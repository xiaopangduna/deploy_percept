#pragma once

#include "deploy_percept/core/post_process/YoloV5DetectPostProcess.hpp"

#include <cstdint>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /** YOLOv5 检测后处理（RKNN int8 三头输出） */
        class YoloV5DetectPostProcessRknn : public YoloV5DetectPostProcess
        {
        public:
            using Params = YoloV5DetectPostProcess::Params;
            using Result = YoloV5DetectPostProcess::Result;

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
