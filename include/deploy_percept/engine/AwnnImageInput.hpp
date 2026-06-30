#pragma once

#ifdef AWNN_FOUND

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/engine/AwnnEngine.hpp"

namespace deploy_percept
{
    namespace engine
    {

        /** VIP 输入 RGB 图像的几何（由 query_input sizes 推断） */
        struct AwnnRgbInputShape
        {
            int width{0};
            int height{0};
            int channels{3};
            std::size_t buffer_bytes{0};
        };

        /** letterbox 参数，便于后处理坐标还原与调试 */
        struct AwnnLetterboxMeta
        {
            float scale{1.f};
            int resize_w{0};
            int resize_h{0};
            int pad_top{0};
            int pad_left{0};
            int pad_bottom{0};
            int pad_right{0};
        };

        /** 从 AwnnEngine::Info 解析输入 0 的 RGB 宽高与 VIP buffer 字节数 */
        AwnnRgbInputShape resolveRgbInputShape(const AwnnEngine::Info &info, std::size_t input_index = 0);

        /**
         * Letterbox + RGB HWC，写入 VIP input buffer（长度 info.input_byte_sizes[0]）。
         * IMAGE_RGB + ADD_PREPROC_NODE 的 .nb 与 model zoo yolov8 一致，buffer 为 interleaved RGB。
         */
        bool prepareLetterboxRgbInput(
            const std::string &input_path,
            const AwnnRgbInputShape &shape,
            cv::Mat &orig_bgr,
            std::vector<std::uint8_t> &input_buffer,
            AwnnLetterboxMeta *letterbox_meta = nullptr);

    } // namespace engine
} // namespace deploy_percept

#endif
