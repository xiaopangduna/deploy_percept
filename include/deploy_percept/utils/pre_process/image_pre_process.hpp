#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/opencv.hpp>
#include <string>

namespace deploy_percept
{
    namespace pre_process
    {

        /**
         * @brief 对图像执行LetterBox预处理，保持宽高比缩放并填充边界
         *
         * @param input_image 输入图像
         * @param target_width 目标宽度
         * @param target_height 目标高度
         * @param pad_color 填充颜色，默认为(128, 128, 128)
         * @param[out] pads_left 左侧填充像素数
         * @param[out] pads_right 右侧填充像素数
         * @param[out] pads_top 顶部填充像素数
         * @param[out] pads_bottom 底部填充像素数
         * @param[out] scale_w 宽度缩放比例
         * @param[out] scale_h 高度缩放比例
         * @return cv::Mat 预处理后的图像
         */
        cv::Mat letterbox_preprocess(
            const cv::Mat &input_image,
            int target_width,
            int target_height,
            cv::Scalar pad_color,
            int &pads_left,
            int &pads_right,
            int &pads_top,
            int &pads_bottom,
            float &scale_w,
            float &scale_h);

        /**
         * @brief 对图像执行LetterBox预处理，保持宽高比缩放并填充边界（带默认参数的重载版本）
         *
         * @param input_image 输入图像
         * @param target_width 目标宽度
         * @param target_height 目标高度
         * @return cv::Mat 预处理后的图像
         */
        cv::Mat letterbox_preprocess(
            const cv::Mat &input_image,
            int target_width,
            int target_height);

        /** AWNN VIP 输入 buffer 线性布局 */
        enum class VipInputPackLayout
        {
            Chw, ///< YOLOv5：[W,H,C,N] + NCHW pack
            Hwc, ///< YOLOv8：[C,H,W,N] + RGB HWC 交错
        };

        /**
         * YOLOv5 AWNN：resize + RGB NCHW pack 至 VIP input buffer（w 最快变）。
         * @p input_buffer 复用：仅当容量不足时 resize。
         */
        bool yolov5_preprocess_awnn(
            const cv::Mat &source_bgr,
            int model_w,
            int model_h,
            int model_c,
            std::vector<std::uint8_t> &input_buffer);

        /**
         * YOLOv8 AWNN：resize + RGB HWC pack 至 VIP input buffer。
         * @p input_buffer 复用：仅当容量不足时 resize。
         */
        bool yolov8_preprocess_awnn(
            const cv::Mat &source_bgr,
            int model_w,
            int model_h,
            std::vector<std::uint8_t> &input_buffer);

    } // namespace pre_process
} // namespace deploy_percept