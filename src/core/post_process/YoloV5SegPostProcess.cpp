#include "deploy_percept/core/post_process/YoloV5SegPostProcess.hpp"
#include <algorithm>

namespace deploy_percept
{
    namespace post_process
    {
        // 构造函数
        YoloV5SegPostProcess::YoloV5SegPostProcess(const Params &params) : params_(params)
        {
            result_.detections.resize(params_.obj_numb_max_size);
            // 初始化分割掩码容器
            result_.label_mask.resize(1);
        }

        bool YoloV5SegPostProcess::run(
            const std::vector<int8_t *> &outputs,
            int input_image_width,
            int input_image_height,
            const std::vector<std::vector<int>> &output_dims,
            const std::vector<float> &output_scales,
            const std::vector<int32_t> &output_zps)
        {
            // ===============================
            // 0. reset state
            // ===============================
            result_.detections.clear();

            if (!result_.label_mask.empty())
            {
                result_.label_mask.clear();
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
                params_.proto_width);

            if (output_dims.size() > 6)
            {
                int8_t *input_proto = (int8_t *)outputs[6]; // 修复：移除多余的解引用操作符*
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

            int grid_h0 = output_dims[0][2];
            int grid_w0 = output_dims[0][3];
            int stride0 = input_image_height / grid_h0;
            validCount += decodeDetectionHead(outputs, 0, params_.anchor_stride8.data(), // 修复：直接传递引用
                                              grid_h0, grid_w0, stride0,
                                              filterBoxes, filterSegments,
                                              objProbs, classId,
                                              params_.conf_threshold,
                                              output_scales, output_zps);

            // stride 16

            int grid_h1 = output_dims[2][2];
            int grid_w1 = output_dims[2][3];
            int stride1 = input_image_height / grid_h1;
            validCount += decodeDetectionHead(outputs, 2, params_.anchor_stride16.data(), // 修复：直接传递引用
                                              grid_h1, grid_w1, stride1,
                                              filterBoxes, filterSegments,
                                              objProbs, classId,
                                              params_.conf_threshold,
                                              output_scales, output_zps);

            // stride 32

            int grid_h2 = output_dims[4][2];
            int grid_w2 = output_dims[4][3];
            int stride2 = input_image_height / grid_h2;
            validCount += decodeDetectionHead(outputs, 4, params_.anchor_stride32.data(), // 修复：直接传递引用
                                              grid_h2, grid_w2, stride2,
                                              filterBoxes, filterSegments,
                                              objProbs, classId,
                                              params_.conf_threshold,
                                              output_scales, output_zps);

            if (validCount <= 0)
            {
                return true;
            }

            // ===============================
            // 3. NMS
            // ===============================
            std::vector<int> indexArray;
            applyNmsToCandidates(
                filterBoxes,
                objProbs,
                classId,
                validCount,
                params_.nms_threshold,
                indexArray);

            int last_count = collectDetectionsWithSegments(
                indexArray,
                filterBoxes,
                objProbs,
                classId,
                filterSegments,
                validCount,
                params_.obj_numb_max_size,
                params_.proto_channel,
                input_image_width,
                input_image_height,
                false,
                result_,
                filterSegments_by_nms);

            if (last_count <= 0)
            {
                return true;
            }

            buildLabelMask(
                result_.detections,
                filterSegments_by_nms,
                proto.data(),
                params_.proto_channel,
                params_.proto_height,
                params_.proto_width,
                input_image_width,
                input_image_height,
                result_.label_mask,
                matmul_out_,
                seg_mask_,
                all_mask_in_one_);

            return true;
        }
        int YoloV5SegPostProcess::decodeDetectionHead(const std::vector<int8_t *> &all_input, int input_id, int *anchor, int grid_h, int grid_w,
                                                      int stride,
                                                      std::vector<float> &boxes, std::vector<float> &segments,
                                                      std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                                                      //   const std::vector<std::vector<int>> &output_dims,
                                                      const std::vector<float> &output_scales,
                                                      const std::vector<int32_t> &output_zps)
        {
            int validCount = 0;
            int grid_len = grid_h * grid_w;

            // 原型掩码 (proto) 现由 run() 函数单独处理

            const int8_t *input = all_input[input_id];         // 修复：直接使用数组访问而不是解引用
            const int8_t *input_seg = all_input[input_id + 1]; // 修复：直接使用数组访问而不是解引用
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
                            const int8_t *in_ptr = input + offset;             // 使用const指针
                            const int8_t *in_ptr_seg = input_seg + offset_seg; // 使用const指针

                            float box_conf_f32 = deqntAffineToF32(box_confidence, zp, scale);

                            if (box_conf_f32 > threshold)
                            {
                                int maxClassId = 0;
                                int8_t maxClassProbs = 0;
                                findV5MaxClass(
                                    in_ptr,
                                    grid_len,
                                    params_.obj_class_num,
                                    maxClassId,
                                    maxClassProbs);

                                const float class_prob_f32 = deqntAffineToF32(maxClassProbs, zp, scale);
                                const float limit_score = box_conf_f32 * class_prob_f32;

                                if (limit_score > threshold)
                                {
                                    float box_x = 0.f;
                                    float box_y = 0.f;
                                    float box_w = 0.f;
                                    float box_h = 0.f;
                                    decodeV5AnchorBox(
                                        in_ptr,
                                        grid_len,
                                        j,
                                        i,
                                        stride,
                                        anchor[a * 2],
                                        anchor[a * 2 + 1],
                                        zp,
                                        scale,
                                        box_x,
                                        box_y,
                                        box_w,
                                        box_h);

                                    for (int k = 0; k < params_.proto_channel; k++)
                                    {
                                        const float seg_element_fp =
                                            deqntAffineToF32(in_ptr_seg[(k)*grid_len], zp_seg, scale_seg);
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

    } // namespace post_process
} // namespace deploy_percept