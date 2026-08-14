/**
 * Sigmastar YOLOv8 分割 demo（SigmastarRuntime + SigmastarEngine + YoloV8SegPostProcessSigmastar）
 *
 * 用法：
 *   yolov8_seg_sigmastar [model.img] [input.jpg] [output.jpg]
 *
 * 默认：./yolov8n_seg_640x640_ssd2355.img  ./bus.jpg  ./yolov8_seg_sigmastar_out.jpg
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
#include "deploy_percept/core/post_process/YoloV8SegPostProcessSigmastar.hpp"
#include "deploy_percept/data/class_labels.hpp"
#include "deploy_percept/utils/vis_draw.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::SigmastarEngine;
using deploy_percept::engine::SigmastarRuntime;
using deploy_percept::post_process::YoloV8SegPostProcessSigmastar;
using deploy_percept::data::kCoco80Labels;
using deploy_percept::utils::drawSegmentationResults;

namespace
{

constexpr const char *kDefaultModel = "yolov8n_seg_640x640_ssd2355.img";
constexpr const char *kDefaultInput = "bus.jpg";
constexpr const char *kDefaultOutput = "yolov8_seg_sigmastar_out.jpg";
constexpr std::uint32_t kDivpYuvAlignment = 16;

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
    cv::Mat &model_input_bgr,
    std::vector<std::uint8_t> &input_buffer)
{
    cv::Mat orig = cv::imread(input_path, cv::IMREAD_COLOR);
    if (orig.empty())
    {
        std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
        return false;
    }

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

} // namespace

int main(int argc, char **argv)
{
    const std::string model_path_str = (argc > 1) ? argv[1] : kDefaultModel;
    const std::string input_path_str = (argc > 2) ? argv[2] : kDefaultInput;
    const char *model_path = model_path_str.c_str();
    const char *input_path = input_path_str.c_str();
    const char *output_path = (argc > 3) ? argv[3] : kDefaultOutput;

    std::printf("yolov8_seg_sigmastar\n");
    std::printf("  model : %s\n", model_path);
    std::printf("  input : %s\n", input_path);
    std::printf("  output: %s\n", output_path);

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

    SigmastarRuntime::Params runtime_params;
    runtime_params.model_paths = {model_path};
    SigmastarRuntime runtime(runtime_params);
    if (!runtime.ok())
    {
        std::fprintf(stderr, "SigmastarRuntime init failed (check IPU firmware)\n");
        return 1;
    }

    SigmastarEngine::Params engine_params;
    engine_params.model_path = model_path;

    SigmastarEngine engine(runtime, engine_params);
    if (!engine.ok())
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
    if (info.output_sizes.size() != 2 || info.output_scalars.size() != 2)
    {
        std::fprintf(
            stderr,
            "expect 2 outputs for yolov8-seg, got shapes=%zu scalars=%zu\n",
            info.output_sizes.size(),
            info.output_scalars.size());
        return 1;
    }

    const int model_h = static_cast<int>(info.input_sizes[0][1]);
    const int model_w = static_cast<int>(info.input_sizes[0][2]);
    const std::size_t buffer_bytes = info.input_byte_sizes[0];

    cv::Mat model_input_bgr;
    std::vector<std::uint8_t> input_buffer;
    if (!prepare_model_input(
            input_path,
            model_w,
            model_h,
            info.input_elm_formats[0],
            buffer_bytes,
            model_input_bgr,
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
    if (tensor_views.size() != 2)
    {
        std::fprintf(stderr, "SigmastarEngine output tensors unavailable\n");
        return 1;
    }

    YoloV8SegPostProcessSigmastar::Params post_params;
    post_params.model_in_w = model_w;
    post_params.model_in_h = model_h;
    post_params.output_shapes = info.output_sizes;
    post_params.output_scalars = info.output_scalars;

    YoloV8SegPostProcessSigmastar processor(post_params);
    if (!processor.run(tensor_views))
    {
        std::fprintf(stderr, "post process failed\n");
        return 1;
    }

    cv::Mat result_img = model_input_bgr.clone();
    drawSegmentationResults(result_img, processor.getResult(), &kCoco80Labels);

    std::error_code ec;
    fs::create_directories(fs::path(output_path).parent_path(), ec);
    if (!cv::imwrite(output_path, result_img))
    {
        std::fprintf(stderr, "failed to save output to %s\n", output_path);
        return 1;
    }

    const auto &seg = processor.getResult();
    std::size_t mask_nnz = 0;
    for (std::uint8_t v : seg.label_mask)
    {
        if (v != 0)
        {
            ++mask_nnz;
        }
    }
    std::printf("detection num: %zu\n", seg.detections.size());
    std::printf("label_mask nnz: %zu / %zu\n", mask_nnz, seg.label_mask.size());
    std::printf("saved seg result to %s\n", output_path);
    return 0;
}
