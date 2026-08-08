/**
 * Sigmastar MobileNetV2 分类 demo（SigmastarEngine + ClassificationPostProcessSigmastar）
 *
 * 用法：
 *   mobilenetv2_classify_sigmastar [model.img] [input.jpg]
 *
 * 默认：./mobilenet_v2_224_ssd2355.img  ./apple.jpg
 *
 * 对齐 OpenDLA classification：INT16 × fScalar → Softmax → Top-1。
 * SDK 开放模型为 5 类：apple / butternut / pizza / goldfish / sea snake。
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "mi_ipu.h"

#include "deploy_percept/core/engine/SigmastarEngine.hpp"
#include "deploy_percept/core/post_process/ClassificationPostProcessSigmastar.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::SigmastarEngine;
using deploy_percept::post_process::ClassificationPostProcessSigmastar;

namespace
{

constexpr const char *kDefaultModel = "mobilenet_v2_224_ssd2355.img";
constexpr const char *kDefaultInput = "apple.jpg";
constexpr std::uint32_t kDivpYuvAlignment = 16;

/** OpenDLA classification demo 标签（与 output class_id 对齐） */
constexpr const char *kOpenDlaClassifyLabels[] = {
    "apple",
    "butternut",
    "pizza",
    "goldfish",
    "sea snake",
};
constexpr int kOpenDlaClassifyLabelCount =
    static_cast<int>(sizeof(kOpenDlaClassifyLabels) / sizeof(kOpenDlaClassifyLabels[0]));

std::uint32_t align_up(std::uint32_t x, std::uint32_t align)
{
    return (x + align - 1u) & ~(align - 1u);
}

std::uint32_t align_down(std::uint32_t x, std::uint32_t align)
{
    return (x / align) * align;
}

/** OpenDLA ST_Algo_YV12toNV12：YV12 (Y+V+U) → NV12 (Y+UV) */
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
    std::vector<std::uint8_t> &input_buffer,
    const char *tag)
{
    if (static_cast<std::size_t>(num_rows) * dst_stride > buffer_bytes)
    {
        std::fprintf(stderr, "%s buffer too small\n", tag);
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
        std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
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
            input_buffer,
            "NV12");
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
            input_buffer,
            "ARGB8888");
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
            input_buffer,
            "ABGR8888");
    }

    std::fprintf(stderr, "unsupported input elm_format: %u\n", elm_format);
    return false;
}

const char *label_name(int class_id, int num_classes)
{
    if (num_classes == kOpenDlaClassifyLabelCount &&
        class_id >= 0 && class_id < kOpenDlaClassifyLabelCount)
    {
        return kOpenDlaClassifyLabels[class_id];
    }
    return nullptr;
}

} // namespace

int main(int argc, char **argv)
{
    const std::string model_path_str = (argc > 1) ? argv[1] : kDefaultModel;
    const std::string input_path_str = (argc > 2) ? argv[2] : kDefaultInput;
    const char *model_path = model_path_str.c_str();
    const char *input_path = input_path_str.c_str();

    std::printf("mobilenetv2_classify_sigmastar\n");
    std::printf("  model : %s\n", model_path);
    std::printf("  input : %s\n", input_path);

    if (!fs::is_regular_file(model_path))
    {
        std::fprintf(stderr, "model not found: %s\n", model_path);
        return 1;
    }
    if (!fs::is_regular_file(input_path))
    {
        std::fprintf(stderr, "input not found: %s\n", input_path);
        return 1;
    }

    SigmastarEngine::Params engine_params;
    engine_params.model_path = model_path;

    SigmastarEngine engine(engine_params);
    if (!engine.is_valid())
    {
        std::fprintf(stderr, "SigmastarEngine init failed\n");
        return 1;
    }

    std::printf("%s", engine.dump_info().c_str());

    const auto &info = engine.getInfo();
    if (info.input_sizes.empty() || info.input_byte_sizes.empty() || info.input_elm_formats.empty())
    {
        std::fprintf(stderr, "SigmastarEngine has no input tensors\n");
        return 1;
    }
    if (info.output_sizes.empty() || info.output_scalars.empty())
    {
        std::fprintf(stderr, "SigmastarEngine has no output tensors\n");
        return 1;
    }

    // IPU shape：常见为 shape[1]=H, shape[2]=W；分类输出 shape[1]=num_classes
    const int model_h = static_cast<int>(info.input_sizes[0][1]);
    const int model_w = static_cast<int>(info.input_sizes[0][2]);
    const std::size_t buffer_bytes = info.input_byte_sizes[0];
    const int num_classes = static_cast<int>(info.output_sizes[0][1]);

    std::vector<std::uint8_t> input_buffer;
    if (!prepare_model_input(
            input_path,
            model_w,
            model_h,
            info.input_elm_formats[0],
            buffer_bytes,
            input_buffer))
    {
        return 1;
    }

    if (!engine.run(input_buffer.data(), input_buffer.size()))
    {
        std::fprintf(stderr, "SigmastarEngine::run failed\n");
        return 1;
    }

    const auto &tensor_views = engine.getResult().tensor_views;
    if (tensor_views.empty())
    {
        std::fprintf(stderr, "SigmastarEngine output tensors unavailable\n");
        return 1;
    }

    ClassificationPostProcessSigmastar processor({
        .num_classes = num_classes,
        .output_scalar = info.output_scalars[0],
    });
    if (!processor.run(tensor_views))
    {
        std::fprintf(stderr, "post process failed\n");
        return 1;
    }

    const auto &result = processor.getResult();
    const char *name = label_name(result.class_id, num_classes);
    if (name != nullptr)
    {
        std::printf(
            "top1: class_id=%d score=%.4f name=%s\n",
            result.class_id,
            result.score,
            name);
    }
    else
    {
        std::printf("top1: class_id=%d score=%.4f\n", result.class_id, result.score);
    }

    return 0;
}
