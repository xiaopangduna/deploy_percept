#include "deploy_percept/core/post_process/YoloDetectPostProcess.hpp"

#include <algorithm>
#include <set>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        void YoloDetectPostProcess::resetResult(Result &result)
        {
            result.group = DetectionResult{};
            result.success = false;
        }

        void YoloDetectPostProcess::applyNmsToCandidates(
            std::vector<float> &candidate_boxes,
            std::vector<float> &candidate_scores,
            std::vector<int> &class_ids,
            int candidate_count,
            float nms_threshold,
            std::vector<int> &index_array)
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
                    nms_threshold);
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
            int last_count = 0;
            result.group.detections.clear();
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

                Detection det{};
                det.box.left = static_cast<int>(clamp(x1, 0, model_in_w));
                det.box.top = static_cast<int>(clamp(y1, 0, model_in_h));
                det.box.right = static_cast<int>(clamp(x2, 0, model_in_w));
                det.box.bottom = static_cast<int>(clamp(y2, 0, model_in_h));
                det.score = obj_conf;
                det.class_id = id;

                result.group.detections.push_back(det);
                ++last_count;
            }

            result.success = (last_count > 0);
            return true;
        }

    } // namespace post_process
} // namespace deploy_percept
