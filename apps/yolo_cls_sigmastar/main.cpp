/**
 * Sigmastar 同进程双模型 demo：YOLOv8 检测 + MobileNetV2 分类。
 *
 * 一个 SigmastarRuntime（CreateDevice 取两模型 var 的 max），两个 SigmastarEngine
 * （各 CreateCHN）。单线程串行 Invoke。
 *
 * 用法：
 *   yolo_cls_sigmastar [--duration-sec T] [yolo.img] [cls.img]
 *                      [detect.jpg] [cls.jpg] [output.jpg]
 *
 * --duration-sec：按墙钟循环（每帧两路各 Invoke 一次），供 percept_monitor 采样
 * 默认模型/图：yolov8n_640x640_ssd2355.img + dog.jpg，
 *             mobilenet_v2_224_ssd2355.img + apple.jpg
 */
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "mi_ipu.h"

#include "deploy_percept/core/engine/SigmastarEngine.hpp"
#include "deploy_percept/core/post_process/ClassificationPostProcessSigmastar.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessSigmastar.hpp"
#include "deploy_percept/data/class_labels.hpp"
#include "deploy_percept/utils/vis_draw.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::SigmastarEngine;
using deploy_percept::engine::SigmastarRuntime;
using deploy_percept::engine::ipu_output_scalars;
using deploy_percept::engine::ipu_output_shapes;
using deploy_percept::post_process::ClassificationPostProcessSigmastar;
using deploy_percept::post_process::YoloV8DetectPostProcessSigmastar;
using deploy_percept::data::kCoco80Labels;
using deploy_percept::data::kImageNet1000Labels;
using deploy_percept::data::lookup_label;
using deploy_percept::utils::drawDetectionResults;

namespace
{

constexpr const char *kDefaultYoloModel = "yolov8n_640x640_ssd2355.img";
constexpr const char *kDefaultClsModel = "mobilenet_v2_224_ssd2355.img";
constexpr const char *kDefaultDetectInput = "dog.jpg";
constexpr const char *kDefaultClsInput = "apple.jpg";
constexpr const char *kDefaultOutput = "yolo_cls_sigmastar_out.jpg";
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

bool require_file(const char *path, const char *what)
{
    if (!fs::is_regular_file(path))
    {
        std::fprintf(stderr, "%s not found: %s\n", what, path);
        return false;
    }
    return true;
}

bool require_input_info(const SigmastarEngine &engine, const char *tag)
{
    if (engine.getInfo().desc.u32InputTensorCount < 1u)
    {
        std::fprintf(stderr, "%s: SigmastarEngine has no input tensors\n", tag);
        return false;
    }
    return true;
}

template <typename Processor>
bool infer_once(
    SigmastarEngine &engine,
    const std::vector<std::uint8_t> &input_buffer,
    Processor &processor,
    const char *tag)
{
    if (!engine.run(input_buffer.data(), input_buffer.size()))
    {
        std::fprintf(stderr, "%s: SigmastarEngine::run failed\n", tag);
        return false;
    }
    const auto &tensor_views = engine.getResult().tensor_views;
    if (tensor_views.empty())
    {
        std::fprintf(stderr, "%s: output tensors unavailable\n", tag);
        return false;
    }
    if (!processor.run(tensor_views))
    {
        std::fprintf(stderr, "%s: post process failed\n", tag);
        return false;
    }
    engine.releaseResult();
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    double duration_sec = 0.0;
    int argi = 1;
    if (argi + 1 < argc && std::strcmp(argv[argi], "--duration-sec") == 0)
    {
        duration_sec = std::atof(argv[argi + 1]);
        if (duration_sec <= 0.0)
        {
            std::fprintf(stderr, "invalid --duration-sec: %s\n", argv[argi + 1]);
            return 1;
        }
        argi += 2;
    }

    const std::string yolo_path_str = (argi < argc) ? argv[argi] : kDefaultYoloModel;
    const std::string cls_path_str = (argi + 1 < argc) ? argv[argi + 1] : kDefaultClsModel;
    const std::string detect_in_str =
        (argi + 2 < argc) ? argv[argi + 2] : kDefaultDetectInput;
    const std::string cls_in_str = (argi + 3 < argc) ? argv[argi + 3] : kDefaultClsInput;
    const char *yolo_path = yolo_path_str.c_str();
    const char *cls_path = cls_path_str.c_str();
    const char *detect_input = detect_in_str.c_str();
    const char *cls_input = cls_in_str.c_str();
    const char *output_path = (argi + 4 < argc) ? argv[argi + 4] : kDefaultOutput;

    std::printf("yolo_cls_sigmastar\n");
    std::printf("  yolo   : %s\n", yolo_path);
    std::printf("  cls    : %s\n", cls_path);
    std::printf("  detect : %s\n", detect_input);
    std::printf("  classify: %s\n", cls_input);
    std::printf("  output : %s\n", output_path);
    if (duration_sec > 0.0)
    {
        std::printf("  duration-sec: %.3f\n", duration_sec);
    }

    if (!require_file(yolo_path, "yolo model") || !require_file(cls_path, "cls model") ||
        !require_file(detect_input, "detect input") || !require_file(cls_input, "cls input"))
    {
        return 1;
    }

    SigmastarRuntime::Params runtime_params;
    runtime_params.model_paths = {yolo_path, cls_path};
    SigmastarRuntime runtime(runtime_params);
    if (!runtime.ok())
    {
        std::fprintf(stderr, "SigmastarRuntime init failed (check IPU firmware)\n");
        return 1;
    }
    std::printf(
        "  CreateDevice var: %u bytes (%.3f MB)\n",
        runtime.params().max_variable_buf_size,
        static_cast<double>(runtime.params().max_variable_buf_size) /
            (1024.0 * 1024.0));

    SigmastarEngine::Params yolo_params;
    yolo_params.model_path = yolo_path;
    SigmastarEngine yolo(runtime, yolo_params);
    if (!yolo.ok())
    {
        std::fprintf(stderr, "YOLOv8 SigmastarEngine init failed\n");
        return 1;
    }

    SigmastarEngine::Params cls_params;
    cls_params.model_path = cls_path;
    SigmastarEngine cls(runtime, cls_params);
    if (!cls.ok())
    {
        std::fprintf(stderr, "MobileNetV2 SigmastarEngine init failed\n");
        return 1;
    }

    if (!require_input_info(yolo, "yolo") || !require_input_info(cls, "cls"))
    {
        return 1;
    }

    const auto &yolo_in = yolo.getInfo().desc.astMI_InputTensorDescs[0];
    const auto &cls_desc = cls.getInfo().desc;
    if (cls_desc.u32OutputTensorCount < 1u)
    {
        std::fprintf(stderr, "cls: SigmastarEngine has no output tensors\n");
        return 1;
    }
    const auto &cls_in = cls_desc.astMI_InputTensorDescs[0];
    const auto &cls_out0 = cls_desc.astMI_OutputTensorDescs[0];

    const int yolo_h = static_cast<int>(yolo_in.u32TensorShape[1]);
    const int yolo_w = static_cast<int>(yolo_in.u32TensorShape[2]);
    const int cls_h = static_cast<int>(cls_in.u32TensorShape[1]);
    const int cls_w = static_cast<int>(cls_in.u32TensorShape[2]);
    const int num_classes = static_cast<int>(cls_out0.u32TensorShape[1]);

    cv::Mat yolo_bgr;
    std::vector<std::uint8_t> yolo_buf;
    if (!prepare_model_input(
            detect_input,
            yolo_w,
            yolo_h,
            static_cast<std::uint32_t>(yolo_in.eElmFormat),
            yolo_in.u32BufSize,
            yolo_bgr,
            yolo_buf))
    {
        return 1;
    }

    cv::Mat cls_bgr;
    std::vector<std::uint8_t> cls_buf;
    if (!prepare_model_input(
            cls_input,
            cls_w,
            cls_h,
            static_cast<std::uint32_t>(cls_in.eElmFormat),
            cls_in.u32BufSize,
            cls_bgr,
            cls_buf))
    {
        return 1;
    }
    (void)cls_bgr;

    YoloV8DetectPostProcessSigmastar::Params yolo_post_params;
    yolo_post_params.model_in_w = yolo_w;
    yolo_post_params.model_in_h = yolo_h;
    yolo_post_params.output_shapes = ipu_output_shapes(yolo.getInfo().desc);
    yolo_post_params.output_scalars = ipu_output_scalars(yolo.getInfo().desc);
    YoloV8DetectPostProcessSigmastar yolo_post(yolo_post_params);

    ClassificationPostProcessSigmastar cls_post({
        .num_classes = num_classes,
        .output_scalar = cls_out0.fScalar,
    });

    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t frames = 0;
    do
    {
        if (!infer_once(yolo, yolo_buf, yolo_post, "yolo") ||
            !infer_once(cls, cls_buf, cls_post, "cls"))
        {
            return 1;
        }
        ++frames;
    } while (duration_sec > 0.0 &&
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() <
                 duration_sec);

    cv::Mat result_img = yolo_bgr.clone();
    drawDetectionResults(result_img, yolo_post.getResult(), &kCoco80Labels);
    std::error_code ec;
    fs::create_directories(fs::path(output_path).parent_path(), ec);
    if (!cv::imwrite(output_path, result_img))
    {
        std::fprintf(stderr, "failed to save output to %s\n", output_path);
        return 1;
    }

    const auto &det = yolo_post.getResult();
    const auto &top1 = cls_post.getResult();
    std::printf("detection num: %zu\n", det.detections.size());
    std::printf("saved detect result to %s\n", output_path);

    const char *cls_name = nullptr;
    if (num_classes == kImageNet1000Labels.num_classes)
    {
        cls_name = lookup_label(kImageNet1000Labels, top1.class_id);
    }
    if (cls_name != nullptr)
    {
        std::printf(
            "top1: class_id=%d score=%.4f name=%s\n",
            top1.class_id,
            top1.score,
            cls_name);
    }
    else
    {
        std::printf("top1: class_id=%d score=%.4f\n", top1.class_id, top1.score);
    }

    if (duration_sec > 0.0)
    {
        const double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf(
            "done: frames=%llu wall=%.3fs\n",
            static_cast<unsigned long long>(frames),
            wall);
    }
    return 0;
}
