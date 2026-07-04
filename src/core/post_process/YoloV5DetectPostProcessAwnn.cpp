#include "deploy_percept/core/post_process/YoloV5DetectPostProcessAwnn.hpp"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        YoloV5DetectPostProcessAwnn::YoloV5DetectPostProcessAwnn(const Params &params)
            : params_(params)
        {
        }

        bool YoloV5DetectPostProcessAwnn::run(const std::vector<TensorView> &inputs)
        {
            // ===============================
            // 0. validate params
            // ===============================
            if (params_.model_in_h <= 0 || params_.model_in_w <= 0)
            {
                return false;
            }

            // ===============================
            // 1. validate inputs
            // ===============================
            if (inputs.size() != 3)
            {
                return false;
            }

            const float *head_ptrs[3]{};
            for (std::size_t i = 0; i < inputs.size(); ++i)
            {
                const TensorView &view = inputs[i];
                if (view.data == nullptr)
                {
                    return false;
                }
                if (view.dtype != TensorDtype::FP32)
                {
                    return false;
                }
                head_ptrs[i] = static_cast<const float *>(view.data);
            }

            resetResult(result_);

            std::vector<float> candidate_boxes;
            std::vector<float> candidate_scores;
            std::vector<int> class_ids;

            // ===============================
            // 2. decode detection heads (stride 8 / 16 / 32)
            // ===============================
            const int valid_count0 = decodeDetectionHeadFp32(
                head_ptrs[0],
                8,
                params_.anchor_stride8,
                candidate_boxes,
                candidate_scores,
                class_ids,
                params_.conf_threshold);

            const int valid_count1 = decodeDetectionHeadFp32(
                head_ptrs[1],
                16,
                params_.anchor_stride16,
                candidate_boxes,
                candidate_scores,
                class_ids,
                params_.conf_threshold);

            const int valid_count2 = decodeDetectionHeadFp32(
                head_ptrs[2],
                32,
                params_.anchor_stride32,
                candidate_boxes,
                candidate_scores,
                class_ids,
                params_.conf_threshold);

            const int candidate_count = valid_count0 + valid_count1 + valid_count2;
            if (candidate_count <= 0)
            {
                return false;
            }

            // ===============================
            // 3. NMS
            // ===============================
            std::vector<int> index_array;
            applyNmsToCandidates(
                candidate_boxes,
                candidate_scores,
                class_ids,
                candidate_count,
                params_.nms_threshold,
                index_array);

            // ===============================
            // 4. collect detections
            // ===============================
            return collectDetections(
                index_array,
                candidate_boxes,
                candidate_scores,
                class_ids,
                candidate_count,
                params_.obj_numb_max_size,
                result_,
                params_.model_in_w,
                params_.model_in_h);
        }

        int YoloV5DetectPostProcessAwnn::decodeDetectionHeadFp32(
            const float *feat,
            int stride,
            const std::vector<int> &anchors,
            std::vector<float> &boxes,
            std::vector<float> &obj_probs,
            std::vector<int> &class_ids,
            float threshold)
        {
            const int model_in_h = params_.model_in_h;
            const int model_in_w = params_.model_in_w;
            const int anchor_num = 3;
            const int feat_w = model_in_w / stride;
            const int feat_h = model_in_h / stride;
            const int cls_num = params_.obj_class_num;

            const float deprob_threshold = YoloBasePostProcess::desigmoid(threshold);
            const int feat_size = feat_w * feat_h;
            const int feat_size_cls_5 = feat_size * (cls_num + 5);

            int valid_count = 0;

            for (int h = 0; h < feat_h; ++h)
            {
                const int h_feat_w_cls_5 = h * feat_w * (cls_num + 5);
                for (int w = 0; w < feat_w; ++w)
                {
                    const int w_cls_5 = w * (cls_num + 5);
                    for (int a = 0; a < anchor_num; ++a)
                    {
                        int class_index = 0;
                        float class_score = -FLT_MAX;
                        const int a_idx = a * feat_size_cls_5 + h_feat_w_cls_5 + w_cls_5;
                        const float *feat_ptr = &feat[a_idx + 4];
                        for (int s = 0; s < cls_num; ++s)
                        {
                            if (*(feat_ptr + s + 1) > class_score)
                            {
                                class_index = s;
                                class_score = *(feat_ptr + s + 1);
                            }
                        }

                        const float box_score = *feat_ptr;
                        float final_score = 0.f;
                        if (box_score >= deprob_threshold && class_score >= deprob_threshold)
                        {
                            final_score = YoloBasePostProcess::sigmoid(box_score) *
                                          YoloBasePostProcess::sigmoid(class_score);
                        }

                        if (final_score >= threshold)
                        {
                            const int loc_idx = a_idx;
                            const float dx = YoloBasePostProcess::sigmoid(feat[loc_idx + 0]);
                            const float dy = YoloBasePostProcess::sigmoid(feat[loc_idx + 1]);
                            const float dw = YoloBasePostProcess::sigmoid(feat[loc_idx + 2]);
                            const float dh = YoloBasePostProcess::sigmoid(feat[loc_idx + 3]);
                            const float pred_cx = (dx * 2.f - 0.5f + w) * stride;
                            const float pred_cy = (dy * 2.f - 0.5f + h) * stride;
                            const float anchor_w = static_cast<float>(anchors[a * 2 + 0]);
                            const float anchor_h = static_cast<float>(anchors[a * 2 + 1]);
                            const float pred_w = dw * dw * 4.f * anchor_w;
                            const float pred_h = dh * dh * 4.f * anchor_h;

                            boxes.push_back(pred_cx - pred_w * 0.5f);
                            boxes.push_back(pred_cy - pred_h * 0.5f);
                            boxes.push_back(pred_w);
                            boxes.push_back(pred_h);
                            obj_probs.push_back(final_score);
                            class_ids.push_back(class_index);
                            ++valid_count;
                        }
                    }
                }
            }

            return valid_count;
        }

    } // namespace post_process
} // namespace deploy_percept
