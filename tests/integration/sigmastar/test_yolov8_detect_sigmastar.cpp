#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "mi_ipu.h"

#include "deploy_percept/core/engine/SigmastarEngine.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessSigmastar.hpp"
#include "deploy_percept/common/types.hpp"
#include "tests/test_common/paths.hpp"
#include "tests/test_common/utils.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::SigmastarEngine;
using deploy_percept::Detection;
using deploy_percept::post_process::YoloV8DetectPostProcessSigmastar;
using percept::test::fixture_data;

namespace
{

constexpr int kBoxTolerancePx = 4;
constexpr std::uint32_t kDivpYuvAlignment = 16;

// Golden：板端 dog.jpg + yolov8n_640x640.img（NV12）
const std::vector<Detection> kExpectedDetections = {
    MakeDetectResult(1, 0.8056f, 102, 151, 474, 467),
    MakeDetectResult(16, 0.7859f, 111, 249, 257, 600),
    MakeDetectResult(2, 0.6287f, 390, 84, 576, 190),
};

std::uint32_t align_up(std::uint32_t x, std::uint32_t align)
{
    return (x + align - 1u) & ~(align - 1u);
}

std::uint32_t align_down(std::uint32_t x, std::uint32_t align)
{
    return (x / align) * align;
}

void yv12_to_nv12(const cv::Mat &yv12, cv::Mat &nv12)
{
    const int width = yv12.cols;
    const int height = yv12.rows * 2 / 3;
    const std::size_t stride = static_cast<std::size_t>(yv12.step[0]);
    yv12.copyTo(nv12);

    cv::Mat in_v(
        height / 2,
        width / 2,
        CV_8UC1,
        yv12.data + stride * static_cast<std::size_t>(height),
        stride / 2);
    cv::Mat in_u(
        height / 2,
        width / 2,
        CV_8UC1,
        yv12.data + stride * static_cast<std::size_t>(height) +
            (stride / 2) * static_cast<std::size_t>(height / 2),
        stride / 2);

    for (int row = 0; row < height / 2; ++row)
    {
        for (int col = 0; col < width / 2; ++col)
        {
            nv12.at<uchar>(height + row, 2 * col) = in_u.at<uchar>(row, col);
            nv12.at<uchar>(height + row, 2 * col + 1) = in_v.at<uchar>(row, col);
        }
    }
}

bool pack_strided_rows(
    const std::uint8_t *src,
    int src_row_bytes,
    int copy_bytes,
    int num_rows,
    std::uint32_t dst_stride,
    std::size_t buffer_bytes,
    std::vector<std::uint8_t> &input_buffer)
{
    if (static_cast<std::size_t>(num_rows) * dst_stride > buffer_bytes)
    {
        return false;
    }
    auto *dst = input_buffer.data();
    for (int row = 0; row < num_rows; ++row)
    {
        std::memcpy(dst, src, static_cast<std::size_t>(copy_bytes));
        dst += dst_stride;
        src += src_row_bytes;
    }
    return true;
}

bool prepare_model_input(
    const std::string &input_path,
    const int model_w,
    const int model_h,
    const std::uint32_t elm_format,
    const std::size_t buffer_bytes,
    std::vector<std::uint8_t> &input_buffer)
{
    cv::Mat orig = cv::imread(input_path, cv::IMREAD_COLOR);
    if (orig.empty())
    {
        return false;
    }

    cv::Mat model_input_bgr;
    cv::resize(orig, model_input_bgr, cv::Size(model_w, model_h));
    input_buffer.assign(buffer_bytes, 0);

    auto fmt = static_cast<MI_IPU_ELEMENT_FORMAT>(elm_format);
    if (fmt == MI_IPU_FORMAT_NV12)
    {
        const int align_h = static_cast<int>(align_down(static_cast<std::uint32_t>(model_h), 2));
        const int align_w = static_cast<int>(align_down(static_cast<std::uint32_t>(model_w), 2));
        cv::Mat align_bgr = model_input_bgr(cv::Range(0, align_h), cv::Range(0, align_w)).clone();
        cv::Mat yv12;
        cv::cvtColor(align_bgr, yv12, cv::COLOR_BGR2YUV_YV12);
        cv::Mat nv12;
        yv12_to_nv12(yv12, nv12);

        const std::uint32_t row_stride =
            align_up(static_cast<std::uint32_t>(model_w), kDivpYuvAlignment);
        return pack_strided_rows(
            nv12.data,
            model_w,
            model_w,
            nv12.rows,
            row_stride,
            buffer_bytes,
            input_buffer);
    }

    if (fmt == MI_IPU_FORMAT_ARGB8888)
    {
        cv::Mat bgra;
        cv::cvtColor(model_input_bgr, bgra, cv::COLOR_BGR2BGRA);
        const std::uint32_t row_stride =
            align_up(static_cast<std::uint32_t>(model_w) * 4u, kDivpYuvAlignment);
        return pack_strided_rows(
            bgra.ptr<std::uint8_t>(0),
            static_cast<int>(bgra.step[0]),
            model_w * 4,
            model_h,
            row_stride,
            buffer_bytes,
            input_buffer);
    }

    if (fmt == MI_IPU_FORMAT_ABGR8888)
    {
        cv::Mat rgba;
        cv::cvtColor(model_input_bgr, rgba, cv::COLOR_BGR2RGBA);
        const std::uint32_t row_stride =
            align_up(static_cast<std::uint32_t>(model_w) * 4u, kDivpYuvAlignment);
        return pack_strided_rows(
            rgba.ptr<std::uint8_t>(0),
            static_cast<int>(rgba.step[0]),
            model_w * 4,
            model_h,
            row_stride,
            buffer_bytes,
            input_buffer);
    }

    return false;
}

void skip_unless_fixture_ready(const fs::path &model_path, const fs::path &input_path)
{
    if (!fs::is_regular_file(model_path))
    {
        GTEST_SKIP() << "model not found: " << model_path;
    }
    if (!fs::is_regular_file(input_path))
    {
        GTEST_SKIP() << "input not found: " << input_path;
    }
}

} // namespace

TEST(YoloV8DetectSigmastarIntegration, DogDetectionsMatchGolden)
{
    const fs::path model_path =
        fixture_data("integration/sigmastar/yolov8_detect/yolov8n_640x640.img");
    const fs::path input_path =
        fixture_data("integration/sigmastar/yolov8_detect/dog.jpg");
    skip_unless_fixture_ready(model_path, input_path);

    SigmastarEngine::Params engine_params;
    engine_params.model_path = model_path.string();

    SigmastarEngine engine(engine_params);
    if (!engine.is_valid())
    {
        GTEST_SKIP() << "SigmastarEngine init failed (need IPU + firmware)";
    }

    const auto &info = engine.getInfo();
    ASSERT_FALSE(info.input_sizes.empty());
    ASSERT_FALSE(info.input_byte_sizes.empty());
    ASSERT_FALSE(info.input_elm_formats.empty());

    const int model_h = static_cast<int>(info.input_sizes[0][1]);
    const int model_w = static_cast<int>(info.input_sizes[0][2]);
    const std::size_t buffer_bytes = info.input_byte_sizes[0];

    std::vector<std::uint8_t> input_buffer;
    ASSERT_TRUE(prepare_model_input(
        input_path.string(),
        model_w,
        model_h,
        info.input_elm_formats[0],
        buffer_bytes,
        input_buffer))
        << "failed to prepare input (elm_format=" << info.input_elm_formats[0] << ")";

    ASSERT_TRUE(engine.run(input_buffer.data(), input_buffer.size()));

    const auto &tensor_views = engine.getResult().tensor_views;
    ASSERT_FALSE(tensor_views.empty());

    YoloV8DetectPostProcessSigmastar::Params post_params;
    post_params.model_in_w = model_w;
    post_params.model_in_h = model_h;
    post_params.output_shapes = info.output_sizes;
    post_params.output_scalars = info.output_scalars;

    YoloV8DetectPostProcessSigmastar processor(post_params);
    ASSERT_TRUE(processor.run(tensor_views));

    expectDetectionsMatchGolden(
        kExpectedDetections,
        processor.getResult().detections,
        kBoxTolerancePx);
}
