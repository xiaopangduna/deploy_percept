#include "deploy_percept/post_process/YoloV8SegPostProcess.hpp"
#include <set>
#include <algorithm>
#include <cmath>
#include <malloc.h>

namespace deploy_percept
{
    namespace post_process
    {
        // 构造函数
        YoloV8SegPostProcess::YoloV8SegPostProcess(const Params &params) : params_(params)
        {
            result_.group.results.resize(params_.obj_numb_max_size);
            // 初始化分割掩码容器
            result_.group.segmentation_masks.resize(1);
        }

        int YoloV8SegPostProcess::decodeDetectionHead(std::vector<void *> *all_input, int input_id, int *anchor, int grid_h, int grid_w,
                                                      int stride,
                                                      std::vector<float> &boxes, std::vector<float> &segments,
                                                      std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                                                      std::vector<std::vector<int>> &output_dims, std::vector<float> &output_scales,
                                                      std::vector<int32_t> &output_zps)
        {
            int validCount = 0;
            int grid_len = grid_h * grid_w;
            int dfl_len = output_dims[0][1] / 4;
            int8_t *box_tensor = (int8_t *)(*all_input)[input_id];
            int32_t box_zp = output_zps[input_id];
            float box_scale = output_scales[input_id];

            int8_t *score_tensor = (int8_t *)(*all_input)[input_id + 1];
            int32_t score_zp = output_zps[input_id + 1];
            float score_scale = output_scales[input_id + 1];
            
            int8_t *score_sum_tensor = nullptr;
            int32_t score_sum_zp = 0;
            float score_sum_scale = 1.0;
            score_sum_tensor = (int8_t *)(*all_input)[input_id + 2];
            score_sum_zp = output_zps[input_id + 2];
            score_sum_scale = output_scales[input_id + 2];

            int8_t *seg_tensor = (int8_t *)(*all_input)[input_id + 3];
            int32_t seg_zp = output_zps[input_id + 3];
            float seg_scale = output_scales[input_id + 3];

            int8_t score_thres_i8 = qntF32ToAffine(threshold, score_zp, score_scale);
            int8_t score_sum_thres_i8 = qntF32ToAffine(threshold, score_sum_zp, score_sum_scale);

            for (int i = 0; i < grid_h; i++)
            {
                for (int j = 0; j < grid_w; j++)
                {
                    int offset = i * grid_w + j;
                    int max_class_id = -1;

                    int offset_seg = i * grid_w + j;
                    int8_t *in_ptr_seg = seg_tensor + offset_seg;

                    // for quick filtering through "score sum"
                    if (score_sum_tensor != nullptr)
                    {
                        if (score_sum_tensor[offset] < score_sum_thres_i8)
                        {
                            continue;
                        }
                    }

                    int8_t max_score = -score_zp;

                    for (int c = 0; c < params_.obj_class_num; c++)
                    {
                        if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                        {
                            max_score = score_tensor[offset];
                            max_class_id = c;
                        }
                        offset += grid_len;
                    }

                    // compute box
                    if (max_score > score_thres_i8)
                    {

                        for (int k = 0; k <params_.proto_channel; k++)
                        {
                            float seg_element_fp = deqntAffineToF32(in_ptr_seg[(k)*grid_len], seg_zp, seg_scale);
                            segments.push_back(seg_element_fp);
                        }

                        offset = i * grid_w + j;
                        float box[4];
                        float before_dfl[dfl_len * 4];
                        for (int k = 0; k < dfl_len * 4; k++)
                        {
                            before_dfl[k] = deqntAffineToF32(box_tensor[offset], box_zp, box_scale);
                            offset += grid_len;
                        }
                        compute_dfl(before_dfl, dfl_len, box);

                        float x1, y1, x2, y2, w, h;
                        x1 = (-box[0] + j + 0.5) * stride;
                        y1 = (-box[1] + i + 0.5) * stride;
                        x2 = (box[2] + j + 0.5) * stride;
                        y2 = (box[3] + i + 0.5) * stride;
                        w = x2 - x1;
                        h = y2 - y1;

                        boxes.push_back(x1);
                        boxes.push_back(y1);
                        boxes.push_back(w);
                        boxes.push_back(h);

                        objProbs.push_back(deqntAffineToF32(max_score, score_zp, score_scale));
                        classId.push_back(max_class_id);
                        validCount++;
                    }
                }
            }

            return validCount;
        }

        bool YoloV8SegPostProcess::run(
            std::vector<void *> *outputs,
            int input_image_width,
            int input_image_height,
            std::vector<std::vector<int>> &output_dims,
            std::vector<float> &output_scales,
            std::vector<int32_t> &output_zps)
        {
            // ===============================
            // 0. reset state
            // ===============================
            result_.group.count = 0;
            result_.success = false;
            result_.message.clear();

            // 删除分割掩码
            result_.group.segmentation_masks.clear();

            std::vector<float> filterBoxes;
            std::vector<float> objProbs;
            std::vector<int> classId;
            std::vector<float> filterSegments;
            std::vector<float> filterSegments_by_nms;

            int validCount = 0;

            // ===============================
            // 1. proto
            // ===============================
            std::vector<float> proto(
                params_.proto_channel *
                params_.proto_height *
                params_.proto_weight);

            if (output_dims.size() > 6)
            {
                int8_t *input_proto = (int8_t *)(*outputs)[12];
                int32_t zp = output_zps[12];
                float scale = output_scales[12];

                std::transform(input_proto, input_proto + proto.size(), proto.begin(),
                               [zp, scale](int8_t val)
                               { return deqntAffineToF32(val, zp, scale); });
            }

            // ===============================
            // 2. detect heads
            // ===============================

            // stride 8
            if (output_dims.size() > 0)
            {
                int grid_h0 = output_dims[0][2];
                int grid_w0 = output_dims[0][3];
                int stride0 = input_image_height / grid_h0;
                validCount += decodeDetectionHead(outputs, 0, params_.anchor_stride8.data(),
                                                  grid_h0, grid_w0, stride0,
                                                  filterBoxes, filterSegments,
                                                  objProbs, classId,
                                                  params_.conf_threshold,
                                                  output_dims, output_scales, output_zps);
            }

            // stride 16
            if (output_dims.size() > 2)
            {
                int grid_h1 = output_dims[4][2];
                int grid_w1 = output_dims[4][3];
                int stride1 = input_image_height / grid_h1;
                validCount += decodeDetectionHead(outputs, 4, params_.anchor_stride16.data(),
                                                  grid_h1, grid_w1, stride1,
                                                  filterBoxes, filterSegments,
                                                  objProbs, classId,
                                                  params_.conf_threshold,
                                                  output_dims, output_scales, output_zps);
            }

            // stride 32
            if (output_dims.size() > 4)
            {
                int grid_h2 = output_dims[8][2];
                int grid_w2 = output_dims[8][3];
                int stride2 = input_image_height / grid_h2;
                validCount += decodeDetectionHead(outputs, 8, params_.anchor_stride32.data(),
                                                  grid_h2, grid_w2, stride2,
                                                  filterBoxes, filterSegments,
                                                  objProbs, classId,
                                                  params_.conf_threshold,
                                                  output_dims, output_scales, output_zps);
            }

            if (validCount <= 0)
            {
                result_.success = true;
                return true;
            }

            // ===============================
            // 3. NMS
            // ===============================
            std::vector<int> indexArray(validCount);
            for (int i = 0; i < validCount; ++i)
                indexArray[i] = i;

            // 使用 std::sort 按 objProbs 降序排列索引
            std::sort(indexArray.begin(), indexArray.end(),
                      [&objProbs](int a, int b)
                      { return objProbs[a] > objProbs[b]; });

            // 对每个类别执行 NMS
            std::set<int> class_set(classId.begin(), classId.end());
            for (int c : class_set)
            {
                retainHighestScoringBoxesByNMS(validCount, filterBoxes, classId, indexArray, c, params_.nms_threshold);
            }

            // 打印NMS后的结果统计信息
            int remaining_count = 0;
            for (int i = 0; i < validCount; ++i) {
                if (indexArray[i] != -1) {
                    remaining_count++;
                }
            }

            int last_count = 0;
            
            // 实现YoloV8版本的collectDetectionsAfterNMS，因为需要填充结果
            for (int i = 0; i < validCount; ++i)
            {
                // 跳过被NMS标记为冗余的检测框，或超出最大检测数量限制的框
                if (indexArray[i] == -1 || last_count >= params_.obj_numb_max_size)
                {
                    continue;
                }

                int n = indexArray[i]; // 获取原始检测框的索引

                // 提取并计算边界框坐标
                float x1 = filterBoxes[n * 4 + 0];      // 左上角x
                float y1 = filterBoxes[n * 4 + 1];      // 左上角y
                float x2 = x1 + filterBoxes[n * 4 + 2]; // 右下角x (x1 + width)
                float y2 = y1 + filterBoxes[n * 4 + 3]; // 右下角y (y1 + height)

                int id = classId[n];          // 保存真实的类别ID
                float obj_conf = objProbs[n]; // 获取该检测框的置信度

                // 收集该检测框对应的分割特征向量
                for (int k = 0; k < params_.proto_channel; k++)
                {
                    filterSegments_by_nms.push_back(filterSegments[n * params_.proto_channel + k]);
                }

                // 填充检测结果结构体（与YOLOv5版本一致，不进行边界限制）
                result_.group.results[last_count].box.left = x1;
                result_.group.results[last_count].box.top = y1;
                result_.group.results[last_count].box.right = x2;
                result_.group.results[last_count].box.bottom = y2;

                result_.group.results[last_count].prop = obj_conf;

                // 设置类别名称
                snprintf(result_.group.results[last_count].name,
                         sizeof(result_.group.results[last_count].name),
                         "class_%d", id);

                // 设置类别ID
                result_.group.results[last_count].cls_id = id;

                last_count++; // 增加有效检测计数
            }

            result_.group.count = last_count;
            
            if (last_count <= 0)
            {
                result_.success = true;
                return true;
            }

            // ===============================
            // 4. boxes / class
            // ===============================
            int boxes_num = last_count;

            std::vector<float> filterBoxes_by_nms(boxes_num * 4);
            std::vector<int> cls_id(boxes_num);

            for (int i = 0; i < boxes_num; ++i)
            {
                filterBoxes_by_nms[i * 4 + 0] = result_.group.results[i].box.left;
                filterBoxes_by_nms[i * 4 + 1] = result_.group.results[i].box.top;
                filterBoxes_by_nms[i * 4 + 2] = result_.group.results[i].box.right;
                filterBoxes_by_nms[i * 4 + 3] = result_.group.results[i].box.bottom;
                cls_id[i] = result_.group.results[i].cls_id;
            }

            // ===============================
            // 5. segmentation (NO malloc)
            // ===============================
            const size_t matmul_size =
                boxes_num * params_.proto_height * params_.proto_weight;
            matmul_out_.assign(matmul_size, 0);

            YoloBasePostProcess::computeSegMask(
                filterSegments_by_nms,
                proto.data(),
                matmul_out_.data(),
                boxes_num,
                params_.proto_channel,
                params_.proto_height * params_.proto_weight);

            const size_t seg_size =
                boxes_num * input_image_height * input_image_width;
            seg_mask_.assign(seg_size, 0);

            YoloBasePostProcess::resizeSegMasks(
                matmul_out_.data(),
                params_.proto_weight,
                params_.proto_height,
                boxes_num,
                seg_mask_.data(),
                input_image_width,
                input_image_height);

            const size_t mask_size =
                input_image_height * input_image_width;
            all_mask_in_one_.assign(mask_size, 0);

            YoloBasePostProcess::mergeBoxMasks(
                seg_mask_.data(),
                all_mask_in_one_.data(),
                filterBoxes_by_nms.data(),
                boxes_num,
                cls_id.data(),
                input_image_height,
                input_image_width);

            // 在处理完所有检测后，分配并填充分割掩码
            if (last_count > 0) {
                const size_t mask_size = input_image_height * input_image_width;
                
                // 为分割掩码分配内存
                result_.group.segmentation_masks.resize(mask_size, 0);
                memcpy(result_.group.segmentation_masks.data(),
                       all_mask_in_one_.data(),
                       mask_size);
            }

            result_.success = true;
            return true;
        }
    } // namespace post_process
} // namespace deploy_percept