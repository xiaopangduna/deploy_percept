#include "deploy_percept/pre_process/ImagePreProcess.hpp"
#include <algorithm>

namespace deploy_percept
{
    namespace pre_process
    {

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
            float &scale_h)
        {
            // 计算缩放比例
            float scale_w_tmp = (float)target_width / input_image.cols;
            float scale_h_tmp = (float)target_height / input_image.rows;
            float min_scale = std::min(scale_w_tmp, scale_h_tmp);

            scale_w = min_scale;
            scale_h = min_scale;

            // 调整图像大小
            cv::Mat resized_image;
            cv::resize(input_image, resized_image, cv::Size(), min_scale, min_scale);

            // 计算填充大小
            int pad_width = target_width - resized_image.cols;
            int pad_height = target_height - resized_image.rows;

            pads_left = pad_width / 2;
            pads_right = pad_width - pads_left;
            pads_top = pad_height / 2;
            pads_bottom = pad_height - pads_top;

            // 在图像周围添加填充
            cv::Mat padded_image;
            cv::copyMakeBorder(resized_image, padded_image,
                               pads_top, pads_bottom,
                               pads_left, pads_right,
                               cv::BORDER_CONSTANT, pad_color);

            return padded_image;
        }

        cv::Mat letterbox_preprocess(
            const cv::Mat &input_image,
            int target_width,
            int target_height)
        {
            int pads_left, pads_right, pads_top, pads_bottom;
            float scale_w, scale_h;

            return letterbox_preprocess(input_image, target_width, target_height,
                                        cv::Scalar(128, 128, 128), pads_left, pads_right,
                                        pads_top, pads_bottom, scale_w, scale_h);
        }



    } // namespace pre_process
} // namespace deploy_percept