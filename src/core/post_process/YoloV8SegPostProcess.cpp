#include "deploy_percept/core/post_process/YoloV8SegPostProcess.hpp"
#include <algorithm>

namespace deploy_percept
{
    namespace post_process
    {
        YoloV8SegPostProcess::YoloV8SegPostProcess(const Params &params) : params_(params)
        {
        }

        bool YoloV8SegPostProcess::run(
            const std::vector<int8_t *> &outputs,
            int input_image_width,
            int input_image_height,
            std::vector<std::vector<int>> &output_dims,
            std::vector<float> &output_scales,
            std::vector<int32_t> &output_zps)
        {
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

            const int8_t *input_proto = outputs[12];
            int32_t zp = output_zps[12];
            float scale = output_scales[12];

            std::transform(input_proto, input_proto + proto.size(), proto.begin(),
                           [zp, scale](int8_t val)
                           { return deqntAffineToF32(val, zp, scale); });

            // ===============================
            // 2. detect heads
            // ===============================

            // stride 8
            int grid_h0 = output_dims[0][2];
            int grid_w0 = output_dims[0][3];
            int stride0 = input_image_height / grid_h0;
            validCount += decodeDetectionAndSegmentationHead(&outputs, 0,
                                                           grid_h0, grid_w0, stride0,
                                                           filterBoxes, filterSegments,
                                                           objProbs, classId,
                                                           params_.conf_threshold,
                                                           16, output_scales, output_zps);

            // stride 16
            int grid_h1 = output_dims[4][2];
            int grid_w1 = output_dims[4][3];
            int stride1 = input_image_height / grid_h1;
            validCount += decodeDetectionAndSegmentationHead(&outputs, 4,
                                                           grid_h1, grid_w1, stride1,
                                                           filterBoxes, filterSegments,
                                                           objProbs, classId,
                                                           params_.conf_threshold,
                                                           16, output_scales, output_zps);

            // stride 32
            int grid_h2 = output_dims[8][2];
            int grid_w2 = output_dims[8][3];
            int stride2 = input_image_height / grid_h2;
            validCount += decodeDetectionAndSegmentationHead(&outputs, 8,
                                                           grid_h2, grid_w2, stride2,
                                                           filterBoxes, filterSegments,
                                                           objProbs, classId,
                                                           params_.conf_threshold,
                                                           16, output_scales, output_zps);

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

            result_.detections.clear();
            result_.label_mask.clear();

            const int last_count = collectDetectionsWithSegments(
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
                true,
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
        int YoloV8SegPostProcess::decodeDetectionAndSegmentationHead(const std::vector<int8_t *> *all_input, int input_id, int grid_h, int grid_w,
                                                                   int stride,
                                                                   std::vector<float> &boxes, std::vector<float> &segments,
                                                                   std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                                                                   int dfl_len, std::vector<float> &output_scales,
                                                                   std::vector<int32_t> &output_zps)
        {
            int validCount = 0;
            int grid_len = grid_h * grid_w;

            const int8_t *box_tensor = (*all_input)[input_id];
            int32_t box_zp = output_zps[input_id];
            float box_scale = output_scales[input_id];

            const int8_t *score_tensor = (*all_input)[input_id + 1];
            int32_t score_zp = output_zps[input_id + 1];
            float score_scale = output_scales[input_id + 1];

            const int8_t *score_sum_tensor = nullptr;
            int32_t score_sum_zp = 0;
            float score_sum_scale = 1.0;
            score_sum_tensor = (*all_input)[input_id + 2];
            score_sum_zp = output_zps[input_id + 2];
            score_sum_scale = output_scales[input_id + 2];

            const int8_t *seg_tensor = (*all_input)[input_id + 3];
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
                    const int8_t *in_ptr_seg = seg_tensor + offset_seg;

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

                        for (int k = 0; k < params_.proto_channel; k++)
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
                        decodeDflBox(before_dfl, dfl_len, box);

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
    } // namespace post_process
} // namespace deploy_percept