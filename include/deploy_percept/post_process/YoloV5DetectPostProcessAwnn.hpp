#pragma once

#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"

#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /** AWNN float 输出 YOLOv5 检测后处理（框坐标在 model 输入尺寸空间） */
        class YoloV5DetectPostProcessAwnn : public YoloV5DetectPostProcess
        {
        public:
            using YoloV5DetectPostProcess::YoloV5DetectPostProcess;

            bool run(
                const std::vector<float *> &inputs,
                int model_in_h,
                int model_in_w);

        private:
            int decodeDetectionHeadFloat(
                const float *input,
                int stride,
                int model_in_h,
                int model_in_w,
                std::vector<float> &boxes,
                std::vector<float> &objProbs,
                std::vector<int> &classId,
                float threshold);
        };

    } // namespace post_process
} // namespace deploy_percept
