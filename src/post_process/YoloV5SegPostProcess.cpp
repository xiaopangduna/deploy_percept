#include "deploy_percept/post_process/YoloV5SegPostProcess.hpp"
#include <set>
#include <algorithm>
#include <cmath>
#include <malloc.h>

namespace deploy_percept
{
    namespace post_process
    {
        // 构造函数
        YoloV5SegPostProcess::YoloV5SegPostProcess(const Params &params) : params_(params)
        {
            result_.group.detection_objects.resize(params_.obj_numb_max_size);
            // 初始化分割掩码容器
            result_.group.segmentation_mask.resize(1);
        }

        int YoloV5SegPostProcess::decodeDetectionHead(std::vector<void *> *all_input, int input_id, int *anchor, int grid_h, int grid_w,
                                                      int stride,
                                                      std::vector<float> &boxes, std::vector<float> &segments,
                                                      std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                                                      std::vector<std::vector<int>> &output_dims, std::vector<float> &output_scales,
                                                      std::vector<int32_t> &output_zps)
        {
            int validCount = 0;
            int grid_len = grid_h * grid_w;

            // 原型掩码 (proto) 现由 run() 函数单独处理

            int8_t *input = (int8_t *)(*all_input)[input_id];
            int8_t *input_seg = (int8_t *)(*all_input)[input_id + 1];
            int32_t zp = output_zps[input_id];
            float scale = output_scales[input_id];
            int32_t zp_seg = output_zps[input_id + 1];
            float scale_seg = output_scales[input_id + 1];

            int8_t thres_i8 = qntF32ToAffine(threshold, zp, scale);

            for (int a = 0; a < 3; a++)
            {
                for (int i = 0; i < grid_h; i++)
                {
                    for (int j = 0; j < grid_w; j++)
                    {
                        int8_t box_confidence = input[((params_.prop_box_size + params_.obj_class_num) * a + 4) * grid_len + i * grid_w + j];
                        if (box_confidence >= thres_i8)
                        {
                            int offset = ((params_.prop_box_size + params_.obj_class_num) * a) * grid_len + i * grid_w + j;
                            int offset_seg = (params_.proto_channel * a) * grid_len + i * grid_w + j;
                            int8_t *in_ptr = input + offset;
                            int8_t *in_ptr_seg = input_seg + offset_seg;

                            float box_conf_f32 = deqntAffineToF32(box_confidence, zp, scale);

                            if (box_conf_f32 > threshold)
                            {
                                // 在条件内部进行类别概率计算和边界框解码
                                int8_t maxClassProbs = in_ptr[5 * grid_len];
                                int maxClassId = 0;
                                for (int k = 1; k < params_.obj_class_num; ++k)
                                {
                                    int8_t prob = in_ptr[(5 + k) * grid_len];
                                    if (prob > maxClassProbs)
                                    {
                                        maxClassId = k;
                                        maxClassProbs = prob;
                                    }
                                }

                                float class_prob_f32 = deqntAffineToF32(maxClassProbs, zp, scale);
                                float limit_score = box_conf_f32 * class_prob_f32;

                                if (limit_score > threshold)
                                {
                                    // 边界框解码
                                    float box_x = (deqntAffineToF32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                                    float box_y = (deqntAffineToF32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                                    float box_w = (deqntAffineToF32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                                    float box_h = (deqntAffineToF32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                                    box_x = (box_x + j) * (float)stride;
                                    box_y = (box_y + i) * (float)stride;
                                    box_w = box_w * box_w * (float)anchor[a * 2];
                                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                                    box_x -= (box_w / 2.0);
                                    box_y -= (box_h / 2.0);

                                    for (int k = 0; k < params_.proto_channel; k++)
                                    {
                                        float seg_element_fp = deqntAffineToF32(in_ptr_seg[(k)*grid_len], zp_seg, scale_seg);
                                        segments.push_back(seg_element_fp);
                                    }

                                    objProbs.push_back(limit_score);
                                    classId.push_back(maxClassId);
                                    validCount++;
                                    boxes.push_back(box_x);
                                    boxes.push_back(box_y);
                                    boxes.push_back(box_w);
                                    boxes.push_back(box_h);
                                }
                            }
                        }
                    }
                }
            }
            return validCount;
        }

        void YoloV5SegPostProcess::collectDetectionsAfterNMS(
            const std::vector<int> &indexArray,
            const std::vector<float> &filterBoxes,
            const std::vector<int> &classId,
            const std::vector<float> &objProbs,
            const std::vector<float> &filterSegments,
            int validCount,
            std::vector<float> &filterSegments_by_nms,
            int &last_count)
        {
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

                // 填充检测结果结构体
                DetectionObject det_obj{};
                det_obj.box.left = x1;
                det_obj.box.top = y1;
                det_obj.box.right = x2;
                det_obj.box.bottom = y2;
                det_obj.prop = obj_conf;
                
                // 设置类别名称
                snprintf(det_obj.name, sizeof(det_obj.name), "class_%d", id);
                
                // 设置类别ID
                det_obj.cls_id = id;

                result_.group.detection_objects.push_back(det_obj);
                last_count++; // 增加有效检测计数
            }
        }

        bool YoloV5SegPostProcess::run(
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
            result_.group.detection_objects.clear(); // 清空之前的检测结果
            result_.success = false;
            result_.message.clear();

            // 清空分割掩码
            if (!result_.group.segmentation_mask.empty()) {
                result_.group.segmentation_mask.clear();
            }

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
                int8_t *input_proto = (int8_t *)(*outputs)[6];
                int32_t zp = output_zps[6];
                float scale = output_scales[6];

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
                int grid_h1 = output_dims[2][2];
                int grid_w1 = output_dims[2][3];
                int stride1 = input_image_height / grid_h1;
                validCount += decodeDetectionHead(outputs, 2, params_.anchor_stride16.data(),
                                                  grid_h1, grid_w1, stride1,
                                                  filterBoxes, filterSegments,
                                                  objProbs, classId,
                                                  params_.conf_threshold,
                                                  output_dims, output_scales, output_zps);
            }

            // stride 32
            if (output_dims.size() > 4)
            {
                int grid_h2 = output_dims[4][2];
                int grid_w2 = output_dims[4][3];
                int stride2 = input_image_height / grid_h2;
                validCount += decodeDetectionHead(outputs, 4, params_.anchor_stride32.data(),
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

            int last_count = 0;
            collectDetectionsAfterNMS(
                indexArray,
                filterBoxes,
                classId,
                objProbs,
                filterSegments,
                validCount,
                filterSegments_by_nms,
                last_count);

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
                filterBoxes_by_nms[i * 4 + 0] = result_.group.detection_objects[i].box.left;
                filterBoxes_by_nms[i * 4 + 1] = result_.group.detection_objects[i].box.top;
                filterBoxes_by_nms[i * 4 + 2] = result_.group.detection_objects[i].box.right;
                filterBoxes_by_nms[i * 4 + 3] = result_.group.detection_objects[i].box.bottom;
                cls_id[i] = result_.group.detection_objects[i].cls_id;
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
                result_.group.segmentation_mask.resize(mask_size, 0);
                memcpy(result_.group.segmentation_mask.data(),
                       all_mask_in_one_.data(),
                       mask_size);
            }

            result_.success = true;
            return true;
        }
    } // namespace post_process
} // namespace deploy_percept