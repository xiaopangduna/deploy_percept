#include "deploy_percept/core/post_process/YoloDetectPostProcess.hpp"

#include <algorithm>
#include <set>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        namespace
        {

            int collectDetectionsFromNms(
                const std::vector<int> &index_array,
                const std::vector<float> &candidate_boxes,
                const std::vector<float> &candidate_scores,
                const std::vector<int> &class_ids,
                int candidate_count,
                int obj_numb_max_size,
                int model_in_w,
                int model_in_h,
                bool clamp_boxes,
                std::vector<Detection> &detections_out,
                const std::vector<float> *segment_coeffs,
                int proto_channel,
                std::vector<float> *segment_coeffs_out)
            {
                detections_out.clear();
                if (segment_coeffs_out != nullptr)
                {
                    segment_coeffs_out->clear();
                }

                int last_count = 0;
                for (int i = 0; i < candidate_count; ++i)
                {
                    if (index_array[i] == -1 || last_count >= obj_numb_max_size)
                    {
                        continue;
                    }

                    const int n = index_array[i];
                    const float x1 = candidate_boxes[n * 4 + 0];
                    const float y1 = candidate_boxes[n * 4 + 1];
                    const float x2 = x1 + candidate_boxes[n * 4 + 2];
                    const float y2 = y1 + candidate_boxes[n * 4 + 3];
                    const int id = class_ids[n];
                    const float obj_conf = candidate_scores[n];

                    if (segment_coeffs != nullptr && segment_coeffs_out != nullptr)
                    {
                        for (int k = 0; k < proto_channel; ++k)
                        {
                            segment_coeffs_out->push_back((*segment_coeffs)[n * proto_channel + k]);
                        }
                    }

                    Detection det{};
                    if (clamp_boxes)
                    {
                        det.box.left = YoloBasePostProcess::clamp(x1, 0, model_in_w);
                        det.box.top = YoloBasePostProcess::clamp(y1, 0, model_in_h);
                        det.box.right = YoloBasePostProcess::clamp(x2, 0, model_in_w);
                        det.box.bottom = YoloBasePostProcess::clamp(y2, 0, model_in_h);
                    }
                    else
                    {
                        det.box.left = static_cast<int>(x1);
                        det.box.top = static_cast<int>(y1);
                        det.box.right = static_cast<int>(x2);
                        det.box.bottom = static_cast<int>(y2);
                    }
                    det.score = obj_conf;
                    det.class_id = id;

                    detections_out.push_back(det);
                    ++last_count;
                }

                return last_count;
            }

        } // namespace

        void YoloDetectPostProcess::resetResult(Result &result)
        {
            result = DetectionResult{};
        }

        void YoloDetectPostProcess::decodeV5AnchorBox(
            const int8_t *in_ptr,
            int grid_len,
            int j,
            int i,
            int stride,
            int anchor_w,
            int anchor_h,
            int32_t zp,
            float scale,
            float &box_x,
            float &box_y,
            float &box_w,
            float &box_h)
        {
            box_x = (deqntAffineToF32(*in_ptr, zp, scale)) * 2.0f - 0.5f;
            box_y = (deqntAffineToF32(in_ptr[grid_len], zp, scale)) * 2.0f - 0.5f;
            box_w = (deqntAffineToF32(in_ptr[2 * grid_len], zp, scale)) * 2.0f;
            box_h = (deqntAffineToF32(in_ptr[3 * grid_len], zp, scale)) * 2.0f;

            box_x = (box_x + static_cast<float>(j)) * static_cast<float>(stride);
            box_y = (box_y + static_cast<float>(i)) * static_cast<float>(stride);
            box_w = box_w * box_w * static_cast<float>(anchor_w);
            box_h = box_h * box_h * static_cast<float>(anchor_h);
            box_x -= (box_w / 2.0f);
            box_y -= (box_h / 2.0f);
        }

        void YoloDetectPostProcess::findV5MaxClass(
            const int8_t *in_ptr,
            int grid_len,
            int obj_class_num,
            int &max_class_id,
            int8_t &max_class_probs)
        {
            max_class_probs = in_ptr[5 * grid_len];
            max_class_id = 0;
            for (int k = 1; k < obj_class_num; ++k)
            {
                const int8_t prob = in_ptr[(5 + k) * grid_len];
                if (prob > max_class_probs)
                {
                    max_class_id = k;
                    max_class_probs = prob;
                }
            }
        }

        void YoloDetectPostProcess::applyNmsToCandidates(
            std::vector<float> &candidate_boxes,
            std::vector<float> &candidate_scores,
            std::vector<int> &class_ids,
            int candidate_count,
            float nms_threshold,
            std::vector<int> &index_array,
            int box_stride)
        {
            index_array.clear();
            index_array.reserve(candidate_count);
            for (int i = 0; i < candidate_count; ++i)
            {
                index_array.push_back(i);
            }

            std::sort(index_array.begin(), index_array.end(),
                      [&candidate_scores](int a, int b) {
                          return candidate_scores[a] > candidate_scores[b];
                      });

            const std::set<int> class_set(class_ids.begin(), class_ids.end());
            for (const int c : class_set)
            {
                retainHighestScoringBoxesByNMS(
                    candidate_count,
                    candidate_boxes,
                    class_ids,
                    index_array,
                    c,
                    nms_threshold,
                    box_stride);
            }
        }

        void YoloDetectPostProcess::applyNmsToProposals(
            const std::vector<DetectionProposal> &proposals,
            float nms_threshold,
            std::vector<int> &picked)
        {
            picked.clear();
            if (proposals.empty())
            {
                return;
            }

            std::vector<int> order(proposals.size());
            for (std::size_t i = 0; i < proposals.size(); ++i)
            {
                order[i] = static_cast<int>(i);
            }

            std::sort(order.begin(), order.end(), [&proposals](int a, int b) {
                return proposals[static_cast<std::size_t>(a)].prob >
                       proposals[static_cast<std::size_t>(b)].prob;
            });

            const int n = static_cast<int>(proposals.size());
            std::vector<float> areas(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i)
            {
                areas[static_cast<std::size_t>(i)] =
                    proposals[static_cast<std::size_t>(i)].w *
                    proposals[static_cast<std::size_t>(i)].h;
            }

            for (int idx : order)
            {
                const auto &a = proposals[static_cast<std::size_t>(idx)];
                int keep = 1;
                for (int kept : picked)
                {
                    const auto &b = proposals[static_cast<std::size_t>(kept)];
                    const float x1 = std::max(a.x, b.x);
                    const float y1 = std::max(a.y, b.y);
                    const float x2 = std::min(a.x + a.w, b.x + b.w);
                    const float y2 = std::min(a.y + a.h, b.y + b.h);
                    const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
                    const float uni =
                        areas[static_cast<std::size_t>(idx)] +
                        areas[static_cast<std::size_t>(kept)] - inter;
                    if (uni > 0.f && inter / uni > nms_threshold)
                    {
                        keep = 0;
                        break;
                    }
                }
                if (keep)
                {
                    picked.push_back(idx);
                }
            }
        }

        bool YoloDetectPostProcess::collectDetections(
            const std::vector<int> &index_array,
            const std::vector<float> &candidate_boxes,
            const std::vector<float> &candidate_scores,
            const std::vector<int> &class_ids,
            int candidate_count,
            int obj_numb_max_size,
            Result &result,
            int model_in_w,
            int model_in_h)
        {
            const int last_count = collectDetectionsFromNms(
                index_array,
                candidate_boxes,
                candidate_scores,
                class_ids,
                candidate_count,
                obj_numb_max_size,
                model_in_w,
                model_in_h,
                true,
                result.detections,
                nullptr,
                0,
                nullptr);

            return last_count > 0;
        }

        int YoloDetectPostProcess::collectDetectionsWithSegments(
            const std::vector<int> &index_array,
            const std::vector<float> &candidate_boxes,
            const std::vector<float> &candidate_scores,
            const std::vector<int> &class_ids,
            const std::vector<float> &segment_coeffs,
            int candidate_count,
            int obj_numb_max_size,
            int proto_channel,
            int model_in_w,
            int model_in_h,
            bool clamp_boxes,
            SegmentationResult &out,
            std::vector<float> &segment_coeffs_out)
        {
            return collectDetectionsFromNms(
                index_array,
                candidate_boxes,
                candidate_scores,
                class_ids,
                candidate_count,
                obj_numb_max_size,
                model_in_w,
                model_in_h,
                clamp_boxes,
                out.detections,
                &segment_coeffs,
                proto_channel,
                &segment_coeffs_out);
        }

    } // namespace post_process
} // namespace deploy_percept
