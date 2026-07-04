#pragma once

#include "deploy_percept/core/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/common/types.hpp"

#include <cstdint>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * YOLO 检测阶段后处理基类（Detect / Seg / Pose 共用）。
         * 提供 Detect 专用 Result 类型与 NMS 等无状态算法；不含 Params、不含 run()，不持有成员。
         */
        class YoloDetectPostProcess : public YoloBasePostProcess
        {
        public:
            using Result = DetectionResult;

            struct DetectionProposal
            {
                float x{0.f};
                float y{0.f};
                float w{0.f};
                float h{0.f};
                int label{0};
                float prob{0.f};
            };

            static void resetResult(Result &result);

            /** box_stride：每候选框占用的 float 数（Detect/Seg 为 4，Pose 为 5） */
            static void applyNmsToCandidates(
                std::vector<float> &candidate_boxes,
                std::vector<float> &candidate_scores,
                std::vector<int> &class_ids,
                int candidate_count,
                float nms_threshold,
                std::vector<int> &index_array,
                int box_stride = 4);

            static void applyNmsToProposals(
                const std::vector<DetectionProposal> &proposals,
                float nms_threshold,
                std::vector<int> &picked);

            static bool collectDetections(
                const std::vector<int> &index_array,
                const std::vector<float> &candidate_boxes,
                const std::vector<float> &candidate_scores,
                const std::vector<int> &class_ids,
                int candidate_count,
                int obj_numb_max_size,
                Result &result,
                int model_in_w,
                int model_in_h);

            static int collectDetectionsWithSegments(
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
                std::vector<float> &segment_coeffs_out);

            /** V5 RKNN anchor 头共用：从量化特征解码 xywh */
            static void decodeV5AnchorBox(
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
                float &box_h);

            /** V5 RKNN anchor 头共用：在 int8 特征上找最大类别 */
            static void findV5MaxClass(
                const int8_t *in_ptr,
                int grid_len,
                int obj_class_num,
                int &max_class_id,
                int8_t &max_class_probs);
        };

    } // namespace post_process
} // namespace deploy_percept
