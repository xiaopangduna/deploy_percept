#pragma once

#include "deploy_percept/core/post_process/YoloDetectPostProcess.hpp"

#include <cstdint>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /** YOLOv5 检测后处理（RKNN int8 三头输出） */
        class YoloV5DetectPostProcessRknn : public YoloDetectPostProcess
        {
        public:
            struct Params
            {
                float conf_threshold = 0.25f;
                float nms_threshold = 0.45f;
                int obj_class_num = 80;
                int obj_numb_max_size = 64;

                std::vector<int> anchor_stride8 = {10, 13, 16, 30, 33, 23};
                std::vector<int> anchor_stride16 = {30, 61, 62, 45, 59, 119};
                std::vector<int> anchor_stride32 = {116, 90, 156, 198, 373, 326};
            };

            using Result = YoloDetectPostProcess::Result;

            explicit YoloV5DetectPostProcessRknn(const Params &params);
            ~YoloV5DetectPostProcessRknn() override = default;

            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(
                const std::vector<int8_t *> &inputs,
                int model_in_h,
                int model_in_w,
                std::vector<int32_t> &qnt_zps,
                std::vector<float> &qnt_scales);

        private:
            int decodeDetectionHead(
                int8_t *input,
                int *anchor,
                int grid_h,
                int grid_w,
                int stride,
                std::vector<float> &boxes,
                std::vector<float> &obj_probs,
                std::vector<int> &class_ids,
                float threshold,
                int32_t zp,
                float scale);

            Params params_;
            Result result_{};
        };

    } // namespace post_process
} // namespace deploy_percept
