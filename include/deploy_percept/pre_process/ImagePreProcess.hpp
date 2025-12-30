#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace deploy_percept {
namespace pre_process {

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
    float &scale_h
);

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
    int target_height
);


} // namespace pre_process
} // namespace deploy_percept