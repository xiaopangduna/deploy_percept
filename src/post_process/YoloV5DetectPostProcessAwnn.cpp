#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

#include <cmath>
#include <cfloat>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        namespace
        {

            float sigmoid(float x)
            {
                return 1.f / (1.f + std::exp(-x));
            }

            float desigmoid(float x)
            {
                return -std::log(1.f / x - 1.f);
            }

        } // namespace

        bool YoloV5DetectPostProcessAwnn::run(
            const std::vector<float *> &inputs,
            int model_in_h,
            int model_in_w)
        {
            if (inputs.size() != 3)
            {
                result_.success = false;
                result_.message = "Invalid input count: expected 3 inputs, got " + std::to_string(inputs.size());
                return false;
            }

            resetResult();

            std::vector<float> filterBoxes;
            std::vector<float> objProbs;
            std::vector<int> classId;

            const int validCount0 = decodeDetectionHeadFloat(
                inputs[0], 8, model_in_h, model_in_w, filterBoxes, objProbs, classId, params_.conf_threshold);
            const int validCount1 = decodeDetectionHeadFloat(
                inputs[1], 16, model_in_h, model_in_w, filterBoxes, objProbs, classId, params_.conf_threshold);
            const int validCount2 = decodeDetectionHeadFloat(
                inputs[2], 32, model_in_h, model_in_w, filterBoxes, objProbs, classId, params_.conf_threshold);

            const int validCount = validCount0 + validCount1 + validCount2;
            return finalizeDetections(filterBoxes, objProbs, classId, validCount, model_in_h, model_in_w);
        }

        int YoloV5DetectPostProcessAwnn::decodeDetectionHeadFloat(
            const float *feat,
            int stride,
            int model_in_h,
            int model_in_w,
            std::vector<float> &boxes,
            std::vector<float> &objProbs,
            std::vector<int> &classId,
            float threshold)
        {
            const int *anchor = nullptr;
            if (stride == 8)
            {
                anchor = params_.anchor_stride8.data();
            }
            else if (stride == 16)
            {
                anchor = params_.anchor_stride16.data();
            }
            else if (stride == 32)
            {
                anchor = params_.anchor_stride32.data();
            }
            else
            {
                return 0;
            }

            const int feat_w = model_in_w / stride;
            const int feat_h = model_in_h / stride;
            const int cls_num = params_.obj_class_num;
            const float deprob_threshold = desigmoid(threshold);

            const int feat_size = feat_w * feat_h;
            const int feat_size_cls_5 = feat_size * (cls_num + 5);

            int validCount = 0;
            for (int h = 0; h < feat_h; ++h)
            {
                const int h_feat_w_cls_5 = h * feat_w * (cls_num + 5);
                for (int w = 0; w < feat_w; ++w)
                {
                    const int w_cls_5 = w * (cls_num + 5);
                    for (int a = 0; a < 3; ++a)
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
                            final_score = sigmoid(box_score) * sigmoid(class_score);
                        }

                        if (final_score < threshold)
                        {
                            continue;
                        }

                        const int loc_idx = a_idx;
                        const float dx = sigmoid(feat[loc_idx + 0]);
                        const float dy = sigmoid(feat[loc_idx + 1]);
                        const float dw = sigmoid(feat[loc_idx + 2]);
                        const float dh = sigmoid(feat[loc_idx + 3]);
                        const float pred_cx = (dx * 2.f - 0.5f + w) * stride;
                        const float pred_cy = (dy * 2.f - 0.5f + h) * stride;
                        const float anchor_w = static_cast<float>(anchor[a * 2 + 0]);
                        const float anchor_h = static_cast<float>(anchor[a * 2 + 1]);
                        const float pred_w = dw * dw * 4.f * anchor_w;
                        const float pred_h = dh * dh * 4.f * anchor_h;
                        const float x0 = pred_cx - pred_w * 0.5f;
                        const float y0 = pred_cy - pred_h * 0.5f;

                        boxes.push_back(x0);
                        boxes.push_back(y0);
                        boxes.push_back(pred_w);
                        boxes.push_back(pred_h);
                        objProbs.push_back(final_score);
                        classId.push_back(class_index);
                        validCount++;
                    }
                }
            }

            return validCount;
        }

    } // namespace post_process
} // namespace deploy_percept
