#pragma once

#include <vector>
#include <cstring>
#include <string>
#include <opencv2/opencv.hpp>
#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"

namespace deploy_percept
{
    namespace post_process
    {
        class YoloV5DetectPostProcess : public YoloBasePostProcess
        {
        public:
            // 参数配置结构体
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

            // 结果结构体
            struct Result
            {
                DetectResultGroup group{};
                bool success = false;
                std::string message{}; // 可选的详细信息
            };

            // 使用参数结构体的构造函数
            explicit YoloV5DetectPostProcess(const Params &params);
            ~YoloV5DetectPostProcess() = default;
            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(
                int8_t *input0,
                int8_t *input1,
                int8_t *input2,
                int model_in_h,
                int model_in_w,
                BoxRect pads,
                float scale_w,
                float scale_h,
                std::vector<int32_t> &qnt_zps,
                std::vector<float> &qnt_scales);

            /**
             * @brief 在图像上绘制检测结果组
             * @param image 输入图像，会在该图像上直接绘制
             * @param detect_result_group 检测结果组，包含所有检测框信息
             * @param font_scale 字体缩放比例，默认为0.4
             * @param line_thickness 线条粗细，默认为3
             */
            void drawDetectionsResultGroupOnImage(cv::Mat& image, 
                                                const DetectResultGroup& detect_result_group,
                                                double font_scale = 0.4,
                                                int line_thickness = 3);

        private:
            Params params_;
            Result result_{};
            void quickSortIndices(std::vector<float> &input, int left, int right, std::vector<int> &indices);

            int processYoloOutput(int8_t *input, int *anchor, int grid_h, int grid_w,
                                  int height, int width, int stride,
                                  std::vector<float> &boxes, std::vector<float> &objProbs,
                                  std::vector<int> &classId, float threshold,
                                  int32_t zp, float scale);
        };

    } // namespace post_process
} // namespace deploy_percept