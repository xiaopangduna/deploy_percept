#pragma once

#include "deploy_percept/core/post_process/YoloV5DetectPostProcess.hpp"

#include <string>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * YOLOv5 检测后处理（AWNN FP32 三头输出，sigmoid 解码）。
         * inputs[0/1/2] 对应 stride 8/16/32，layout 与 ai-sdk yolov5.nb 一致。
         */
        class YoloV5DetectPostProcessAwnn : public YoloV5DetectPostProcess
        {
        public:
            struct Params : YoloV5DetectPostProcess::Params
            {
                /** 模型输入高宽（VIP 顺序 [W,H,C,N]） */
                int model_in_h{0};
                int model_in_w{0};
            };

            using Result = YoloV5DetectPostProcess::Result;

            explicit YoloV5DetectPostProcessAwnn(const Params &params);
            ~YoloV5DetectPostProcessAwnn() override = default;

            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(const std::vector<TensorView> &inputs);

        private:
            int decodeDetectionHeadFp32(
                const float *feat,
                int stride,
                const std::vector<int> &anchors,
                std::vector<float> &boxes,
                std::vector<float> &obj_probs,
                std::vector<int> &class_ids,
                float threshold);

            static float sigmoid(float x);
            static float desigmoid(float x);

            Params params_;
            Result result_{};
        };

    } // namespace post_process
} // namespace deploy_percept
