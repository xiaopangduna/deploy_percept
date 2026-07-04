#pragma once

#include "deploy_percept/core/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/common/types.hpp"

#include <string>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * YOLOv5 目标检测后处理基类（V5 Detect 系列共用）。
         * 仅提供 Params/Result 类型与无状态算法；不含 run()，不持有成员。
         * RKNN / AWNN 输入与 decode 由子类实现，状态（params_/result_）只在叶子类。
         */
        class YoloV5DetectPostProcess : public YoloBasePostProcess
        {
        public:
            struct Params
            {
                float conf_threshold = 0.25f;
                float nms_threshold = 0.45f;
                int obj_class_num = 80;
                int obj_name_max_size = 16;
                int obj_numb_max_size = 64;

                std::vector<int> anchor_stride8 = {10, 13, 16, 30, 33, 23};
                std::vector<int> anchor_stride16 = {30, 61, 62, 45, 59, 119};
                std::vector<int> anchor_stride32 = {116, 90, 156, 198, 373, 326};
            };

            struct Result
            {
                ResultGroup group{};
                bool success = false;
                std::string message{};
            };

            ~YoloV5DetectPostProcess() override = default;

            static void resetResult(Result &result);

            /** 按置信度排序并对候选框执行按类 NMS，结果写入 index_array（-1 表示被抑制）。 */
            static void applyNmsToCandidates(
                std::vector<float> &candidate_boxes,
                std::vector<float> &candidate_scores,
                std::vector<int> &class_ids,
                int candidate_count,
                float nms_threshold,
                std::vector<int> &index_array);

            /** 将 NMS 保留的候选框转为 DetectionObject 并写入 result.group。 */
            static bool collectDetections(
                const std::vector<int> &index_array,
                const std::vector<float> &candidate_boxes,
                const std::vector<float> &candidate_scores,
                const std::vector<int> &class_ids,
                int candidate_count,
                int obj_numb_max_size,
                int obj_name_max_size,
                Result &result,
                int model_in_w,
                int model_in_h);
        };

    } // namespace post_process
} // namespace deploy_percept
