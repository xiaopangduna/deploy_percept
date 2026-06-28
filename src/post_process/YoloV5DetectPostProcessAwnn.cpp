#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <set>
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

        YoloV5DetectPostProcessAwnn::YoloV5DetectPostProcessAwnn(const Params &params)
            : YoloBasePostProcess(), params_(params)
        {
        }

        void YoloV5DetectPostProcessAwnn::resetResult()
        {
            result_.group = ResultGroup{};
            result_.success = false;
            result_.message.clear();
        }

        bool YoloV5DetectPostProcessAwnn::finalizeDetections(
            std::vector<float> &filterBoxes,
            std::vector<float> &objProbs,
            std::vector<int> &classId,
            int validCount)
        {
            const int model_in_h = params_.model_in_h;
            const int model_in_w = params_.model_in_w;
            if (validCount <= 0)
            {
                result_.message = "No objects detected";
                return false;
            }

            std::vector<int> indexArray;
            indexArray.reserve(validCount);
            for (int i = 0; i < validCount; ++i)
            {
                indexArray.push_back(i);
            }

            YoloBasePostProcess::quickSortIndices(objProbs, 0, validCount - 1, indexArray);

            const std::set<int> class_set(classId.begin(), classId.end());
            for (const int c : class_set)
            {
                retainHighestScoringBoxesByNMS(
                    validCount, filterBoxes, classId, indexArray, c, params_.nms_threshold);
            }

            int last_count = 0;
            result_.group.count = 0;
            result_.group.detection_objects.clear();
            for (int i = 0; i < validCount; ++i)
            {
                if (indexArray[i] == -1 || last_count >= params_.obj_numb_max_size)
                {
                    continue;
                }
                const int n = indexArray[i];

                const float x1 = filterBoxes[n * 4 + 0];
                const float y1 = filterBoxes[n * 4 + 1];
                const float x2 = x1 + filterBoxes[n * 4 + 2];
                const float y2 = y1 + filterBoxes[n * 4 + 3];

                const int id = classId[n];
                const float obj_conf = objProbs[n];

                DetectionObject det_obj{};
                det_obj.box.left = static_cast<int>(clamp(x1, 0, model_in_w));
                det_obj.box.top = static_cast<int>(clamp(y1, 0, model_in_h));
                det_obj.box.right = static_cast<int>(clamp(x2, 0, model_in_w));
                det_obj.box.bottom = static_cast<int>(clamp(y2, 0, model_in_h));
                det_obj.prop = obj_conf;
                det_obj.cls_id = id;

                std::snprintf(det_obj.name, params_.obj_name_max_size, "class_%d", id);

                result_.group.detection_objects.push_back(det_obj);
                ++last_count;
            }
            result_.group.count = last_count;

            result_.success = (last_count > 0);
            result_.message = "Processing completed successfully";
            return true;
        }

        int YoloV5DetectPostProcessAwnn::decodeDetectionHeadFp32(
            const float *feat,
            int stride,
            const std::vector<int> &anchors,
            std::vector<float> &boxes,
            std::vector<float> &objProbs,
            std::vector<int> &classId,
            float threshold)
        {
            const int model_in_h = params_.model_in_h;
            const int model_in_w = params_.model_in_w;
            const int anchor_num = 3;
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
                            final_score = sigmoid(box_score) * sigmoid(class_score);
                        }

                        if (final_score >= threshold)
                        {
                            const int loc_idx = a_idx;
                            const float dx = sigmoid(feat[loc_idx + 0]);
                            const float dy = sigmoid(feat[loc_idx + 1]);
                            const float dw = sigmoid(feat[loc_idx + 2]);
                            const float dh = sigmoid(feat[loc_idx + 3]);
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
                            objProbs.push_back(final_score);
                            classId.push_back(class_index);
                            ++validCount;
                        }
                    }
                }
            }

            return validCount;
        }

        bool YoloV5DetectPostProcessAwnn::run(const std::vector<TensorView> &inputs)
        {
            if (params_.model_in_h <= 0 || params_.model_in_w <= 0)
            {
                result_.success = false;
                result_.message = "Invalid Params: model_in_h and model_in_w must be positive";
                return false;
            }

            if (inputs.size() != 3)
            {
                result_.success = false;
                result_.message =
                    "Invalid input count: expected 3 inputs, got " + std::to_string(inputs.size());
                return false;
            }

            const float *head_ptrs[3]{};
            for (std::size_t i = 0; i < inputs.size(); ++i)
            {
                const TensorView &view = inputs[i];
                if (view.data == nullptr)
                {
                    result_.success = false;
                    result_.message = "Input tensor " + std::to_string(i) + " has null data";
                    return false;
                }
                if (view.dtype != TensorDtype::FP32)
                {
                    result_.success = false;
                    result_.message = "Input tensor " + std::to_string(i) + " is not FP32";
                    return false;
                }
                head_ptrs[i] = static_cast<const float *>(view.data);
            }

            resetResult();

            std::vector<float> filterBoxes;
            std::vector<float> objProbs;
            std::vector<int> classId;

            const int validCount0 = decodeDetectionHeadFp32(
                head_ptrs[0],
                8,
                params_.anchor_stride8,
                filterBoxes,
                objProbs,
                classId,
                params_.conf_threshold);

            const int validCount1 = decodeDetectionHeadFp32(
                head_ptrs[1],
                16,
                params_.anchor_stride16,
                filterBoxes,
                objProbs,
                classId,
                params_.conf_threshold);

            const int validCount2 = decodeDetectionHeadFp32(
                head_ptrs[2],
                32,
                params_.anchor_stride32,
                filterBoxes,
                objProbs,
                classId,
                params_.conf_threshold);

            const int validCount = validCount0 + validCount1 + validCount2;
            return finalizeDetections(filterBoxes, objProbs, classId, validCount);
        }

    } // namespace post_process
} // namespace deploy_percept
