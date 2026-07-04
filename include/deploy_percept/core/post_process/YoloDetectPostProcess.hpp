#pragma once

#include "deploy_percept/core/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/common/types.hpp"

#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * YOLO 检测阶段后处理基类（Detect / Seg / Pose 共用）。
         * 提供 Detect 专用 Result 类型与 NMS 等无状态算法；不含 Params、不含 run()，不持有成员。
         * Detect 叶子可用 Result；Seg / Pose 叶子自行定义 Result，仅复用 static 方法。
         */
        class YoloDetectPostProcess : public YoloBasePostProcess
        {
        public:
            /** Detect 叶子类共用的结果类型 */
            struct Result
            {
                DetectionResult group{};
                bool success = false;
            };

            /** (x, y, w, h) 候选框，供 V8 DFL decode 与 NMS 使用 */
            struct DetectionProposal
            {
                float x{0.f};
                float y{0.f};
                float w{0.f};
                float h{0.f};
                int label{0};
                float prob{0.f};
            };

            ~YoloDetectPostProcess() override = default;

            static void resetResult(Result &result);

            /** 按置信度排序并对候选框执行按类 NMS，结果写入 index_array（-1 表示被抑制） */
            static void applyNmsToCandidates(
                std::vector<float> &candidate_boxes,
                std::vector<float> &candidate_scores,
                std::vector<int> &class_ids,
                int candidate_count,
                float nms_threshold,
                std::vector<int> &index_array);

            /** 对 (x,y,w,h) proposals 做 class-agnostic greedy NMS，picked 为保留的下标（按 score 序） */
            static void applyNmsToProposals(
                const std::vector<DetectionProposal> &proposals,
                float nms_threshold,
                std::vector<int> &picked);

            /** 将 NMS 保留的候选框转为 Detection 并写入 result.group */
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
        };

    } // namespace post_process
} // namespace deploy_percept
