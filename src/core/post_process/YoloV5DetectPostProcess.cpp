#include "deploy_percept/core/post_process/YoloV5DetectPostProcess.hpp"

#include <cstdio>
#include <set>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        void YoloV5DetectPostProcess::resetResult(Result &result)
        {
            result.group = ResultGroup{};
            result.success = false;
            result.message.clear();
        }

        void YoloV5DetectPostProcess::applyNmsToCandidates(
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

            YoloBasePostProcess::quickSortIndices(
                candidate_scores, 0, candidate_count - 1, index_array);

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

        bool YoloV5DetectPostProcess::collectDetections(
            const std::vector<int> &index_array,
            const std::vector<float> &candidate_boxes,
            const std::vector<float> &candidate_scores,
            const std::vector<int> &class_ids,
            int candidate_count,
            int obj_numb_max_size,
            int obj_name_max_size,
            Result &result,
            int model_in_w,
            int model_in_h)
        {
            int last_count = 0;
            result.group.count = 0;
            result.group.detection_objects.clear();
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

                DetectionObject det_obj{};
                det_obj.box.left = static_cast<int>(clamp(x1, 0, model_in_w));
                det_obj.box.top = static_cast<int>(clamp(y1, 0, model_in_h));
                det_obj.box.right = static_cast<int>(clamp(x2, 0, model_in_w));
                det_obj.box.bottom = static_cast<int>(clamp(y2, 0, model_in_h));
                det_obj.prop = obj_conf;
                det_obj.cls_id = id;

                std::snprintf(det_obj.name, obj_name_max_size, "class_%d", id);

                result.group.detection_objects.push_back(det_obj);
                ++last_count;
            }
            result.group.count = last_count;

            result.success = (last_count > 0);
            result.message = "Processing completed successfully";
            return true;
        }

    } // namespace post_process
} // namespace deploy_percept
