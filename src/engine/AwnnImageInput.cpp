#ifdef AWNN_FOUND

#include "deploy_percept/engine/AwnnImageInput.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace deploy_percept
{
    namespace engine
    {

        namespace
        {

            int indexOfChannelDim(const std::array<std::uint32_t, 6> &sizes, std::uint32_t num_dims)
            {
                for (std::uint32_t i = 0; i < num_dims; ++i)
                {
                    if (sizes[i] == 3)
                    {
                        return static_cast<int>(i);
                    }
                }
                return -1;
            }

        } // namespace

        AwnnRgbInputShape resolveRgbInputShape(const AwnnEngine::Info &info, const std::size_t input_index)
        {
            AwnnRgbInputShape shape{};
            shape.buffer_bytes = info.input_byte_sizes.at(input_index);
            const auto &sizes = info.input_sizes.at(input_index);
            const auto num_dims = info.input_num_dims.at(input_index);
            const int c_idx = indexOfChannelDim(sizes, num_dims);

            if (num_dims >= 4 && sizes[0] == 1 && c_idx == 3)
            {
                // NHWC [1, H, W, 3]
                shape.height = static_cast<int>(sizes[1]);
                shape.width = static_cast<int>(sizes[2]);
            }
            else if (num_dims >= 4 && sizes[0] == 1 && c_idx == 1)
            {
                // 网络内部 NCHW [1, 3, H, W]；IMAGE_RGB 输入仍按 H×W letterbox
                shape.height = static_cast<int>(sizes[2]);
                shape.width = static_cast<int>(sizes[3]);
            }
            else if (num_dims >= 4 && c_idx == 2 && sizes[3] == 1)
            {
                // VIP WHCN [W, H, C, N]
                shape.width = static_cast<int>(sizes[0]);
                shape.height = static_cast<int>(sizes[1]);
            }
            else if (num_dims >= 3 && c_idx == 2)
            {
                // VIP [W, H, 3] 或 [H, W, 3]；优先 WHCN（与 AwnnEngine 注释一致）
                shape.width = static_cast<int>(sizes[0]);
                shape.height = static_cast<int>(sizes[1]);
            }
            else if (c_idx == 0 && num_dims >= 3)
            {
                // CHW [3, H, W]
                shape.height = static_cast<int>(sizes[1]);
                shape.width = static_cast<int>(sizes[2]);
            }
            else
            {
                shape.width = static_cast<int>(sizes[0]);
                shape.height = static_cast<int>(sizes[1]);
            }

            if (c_idx >= 0)
            {
                shape.channels = static_cast<int>(sizes[static_cast<std::size_t>(c_idx)]);
            }

            return shape;
        }

        bool prepareLetterboxRgbInput(
            const std::string &input_path,
            const AwnnRgbInputShape &shape,
            cv::Mat &orig_bgr,
            std::vector<std::uint8_t> &input_buffer,
            AwnnLetterboxMeta *letterbox_meta)
        {
            if (shape.width <= 0 || shape.height <= 0 || shape.buffer_bytes == 0)
            {
                return false;
            }

            orig_bgr = cv::imread(input_path, cv::IMREAD_COLOR);
            if (orig_bgr.empty())
            {
                return false;
            }

            cv::Mat rgb;
            cv::cvtColor(orig_bgr, rgb, cv::COLOR_BGR2RGB);

            float scale_letterbox = 1.f;
            if ((shape.height * 1.f / rgb.rows) < (shape.width * 1.f / rgb.cols))
            {
                scale_letterbox = shape.height * 1.f / rgb.rows;
            }
            else
            {
                scale_letterbox = shape.width * 1.f / rgb.cols;
            }

            const int resize_cols = static_cast<int>(std::round(scale_letterbox * rgb.cols));
            const int resize_rows = static_cast<int>(std::round(scale_letterbox * rgb.rows));
            cv::resize(rgb, rgb, cv::Size(resize_cols, resize_rows));

            input_buffer.assign(shape.buffer_bytes, 0);

            // 与 model zoo yolov8_6_pre.cpp 一致
            const float dh = static_cast<float>(shape.height - resize_rows) / 2.f;
            const float dw = static_cast<float>(shape.width - resize_cols) / 2.f;
            const int top = static_cast<int>(std::round(dh - 0.1f));
            const int bot = static_cast<int>(std::round(dh + 0.1f));
            const int left = static_cast<int>(std::round(dw - 0.1f));
            const int right = static_cast<int>(std::round(dw + 0.1f));

            cv::Mat letterboxed(shape.height, shape.width, CV_8UC3, input_buffer.data());
            cv::copyMakeBorder(
                rgb,
                letterboxed,
                top,
                bot,
                left,
                right,
                cv::BORDER_CONSTANT,
                cv::Scalar(114, 114, 114));

            if (letterbox_meta != nullptr)
            {
                letterbox_meta->scale = scale_letterbox;
                letterbox_meta->resize_w = resize_cols;
                letterbox_meta->resize_h = resize_rows;
                letterbox_meta->pad_top = top;
                letterbox_meta->pad_left = left;
                letterbox_meta->pad_bottom = bot;
                letterbox_meta->pad_right = right;
            }

            return true;
        }

    } // namespace engine
} // namespace deploy_percept

#endif
