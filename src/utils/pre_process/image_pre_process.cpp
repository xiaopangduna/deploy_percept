#include "deploy_percept/utils/pre_process/image_pre_process.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace deploy_percept
{
    namespace pre_process
    {

        namespace
        {

            void pack_rgb_nchw_into(
                const cv::Mat &bgr,
                const int width,
                const int height,
                const int channels,
                std::vector<std::uint8_t> &buffer)
            {
                cv::Mat rgb;
                cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

                const std::size_t byte_size =
                    static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                    static_cast<std::size_t>(channels);
                if (buffer.size() != byte_size)
                {
                    buffer.resize(byte_size);
                }

                for (int h = 0; h < height; ++h)
                {
                    for (int w = 0; w < width; ++w)
                    {
                        const cv::Vec3b pix = rgb.at<cv::Vec3b>(h, w);
                        for (int c = 0; c < channels; ++c)
                        {
                            buffer[c * height * width + h * width + w] = pix[c];
                        }
                    }
                }
            }

            void pack_rgb_hwc_into(
                const cv::Mat &bgr,
                const int width,
                const int height,
                std::vector<std::uint8_t> &buffer)
            {
                cv::Mat rgb;
                cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

                const std::size_t byte_size =
                    static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
                if (buffer.size() != byte_size)
                {
                    buffer.resize(byte_size);
                }

                cv::Mat rgb_hwc(height, width, CV_8UC3, buffer.data());
                rgb.copyTo(rgb_hwc);
            }

        } // namespace

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
            float scale_w_tmp = static_cast<float>(target_width) / input_image.cols;
            float scale_h_tmp = static_cast<float>(target_height) / input_image.rows;
            float min_scale = std::min(scale_w_tmp, scale_h_tmp);

            scale_w = min_scale;
            scale_h = min_scale;

            cv::Mat resized_image;
            cv::resize(input_image, resized_image, cv::Size(), min_scale, min_scale);

            int pad_width = target_width - resized_image.cols;
            int pad_height = target_height - resized_image.rows;

            pads_left = pad_width / 2;
            pads_right = pad_width - pads_left;
            pads_top = pad_height / 2;
            pads_bottom = pad_height - pads_top;

            cv::Mat padded_image;
            cv::copyMakeBorder(
                resized_image,
                padded_image,
                pads_top,
                pads_bottom,
                pads_left,
                pads_right,
                cv::BORDER_CONSTANT,
                pad_color);

            return padded_image;
        }

        cv::Mat letterbox_preprocess(
            const cv::Mat &input_image,
            const int target_width,
            const int target_height)
        {
            int pads_left = 0;
            int pads_right = 0;
            int pads_top = 0;
            int pads_bottom = 0;
            float scale_w = 0.f;
            float scale_h = 0.f;

            return letterbox_preprocess(
                input_image,
                target_width,
                target_height,
                cv::Scalar(128, 128, 128),
                pads_left,
                pads_right,
                pads_top,
                pads_bottom,
                scale_w,
                scale_h);
        }

        bool yolov5_preprocess_awnn(
            const cv::Mat &source_bgr,
            const int model_w,
            const int model_h,
            const int model_c,
            std::vector<std::uint8_t> &input_buffer)
        {
            if (source_bgr.empty())
            {
                return false;
            }

            cv::Mat resized;
            cv::resize(source_bgr, resized, cv::Size(model_w, model_h));
            pack_rgb_nchw_into(resized, model_w, model_h, model_c, input_buffer);
            return true;
        }

        bool yolov8_preprocess_awnn(
            const cv::Mat &source_bgr,
            const int model_w,
            const int model_h,
            std::vector<std::uint8_t> &input_buffer)
        {
            if (source_bgr.empty())
            {
                return false;
            }

            cv::Mat resized;
            cv::resize(source_bgr, resized, cv::Size(model_w, model_h));
            pack_rgb_hwc_into(resized, model_w, model_h, input_buffer);
            return true;
        }

    } // namespace pre_process
} // namespace deploy_percept
