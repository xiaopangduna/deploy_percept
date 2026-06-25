#pragma once

#include <string>
#include <vector>

#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * YOLOv5 检测后处理（AWNN FP32 三头输出，sigmoid 解码）。
         * inputs[0/1/2] 对应 stride 8/16/32，layout 与 ai-sdk yolov5.nb 一致。
         */
        class YoloV5DetectPostProcessAwnn : public YoloBasePostProcess
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

            explicit YoloV5DetectPostProcessAwnn(const Params &params);
            ~YoloV5DetectPostProcessAwnn() = default;

            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(
                const std::vector<float *> &inputs,
                int model_in_h,
                int model_in_w);

        private:
            void resetResult();

            bool finalizeDetections(
                std::vector<float> &filterBoxes,
                std::vector<float> &objProbs,
                std::vector<int> &classId,
                int validCount,
                int model_in_h,
                int model_in_w);

            int decodeDetectionHeadFp32(
                const float *feat,
                int stride,
                const std::vector<int> &anchors,
                int model_in_h,
                int model_in_w,
                std::vector<float> &boxes,
                std::vector<float> &objProbs,
                std::vector<int> &classId,
                float threshold);

            Params params_;
            Result result_{};
        };

    } // namespace post_process
} // namespace deploy_percept
