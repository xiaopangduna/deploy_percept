/**
 * Allwinner 同进程双模型 demo：YOLOv8 检测 + MobileNetV2 分类。
 *
 * 一个 VipLiteRuntime，两个 AwnnEngine。主线程每帧串行 yolo → cls。
 *
 * 用法：
 *   yolo_cls_awnn [--duration-sec T] [yolo.nb] [cls.nb]
 *                 [detect.jpg] [cls.jpg] [output.jpg]
 *
 * --duration-sec：按墙钟循环（每帧两路各 run 一次），供 percept_monitor 采样
 * 默认：./yolov8_a733.nb + ./dog.jpg，./mobilenetv2_a733.nb + ./cat.jpg
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

#include "deploy_percept/core/engine/AwnnEngine.hpp"
#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessAwnn.hpp"
#include "deploy_percept/data/class_labels.hpp"
#include "deploy_percept/utils/vis_draw.hpp"
#include "deploy_percept/version.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::ClassificationPostProcessAwnn;
using deploy_percept::post_process::YoloV8DetectPostProcessAwnn;
using deploy_percept::data::kCoco80Labels;
using deploy_percept::data::kImageNet1000Labels;
using deploy_percept::data::lookup_label;
using deploy_percept::utils::drawDetectionResults;

namespace
{

constexpr const char *kDefaultYoloModel = "yolov8_a733.nb";
constexpr const char *kDefaultClsModel = "mobilenetv2_a733.nb";
constexpr const char *kDefaultDetectInput = "dog.jpg";
constexpr const char *kDefaultClsInput = "cat.jpg";
constexpr const char *kDefaultOutput = "yolo_cls_awnn_out.jpg";

bool require_file(const char *path, const char *what)
{
    if (!fs::is_regular_file(path))
    {
        std::fprintf(stderr, "%s not found: %s\n", what, path);
        return false;
    }
    return true;
}

bool prepare_model_input(
    const std::string &input_path,
    const int model_w,
    const int model_h,
    const std::size_t buffer_bytes,
    cv::Mat &model_input,
    std::vector<std::uint8_t> &input_buffer)
{
    cv::Mat orig = cv::imread(input_path, cv::IMREAD_COLOR);
    if (orig.empty())
    {
        std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
        return false;
    }

    cv::resize(orig, model_input, cv::Size(model_w, model_h));
    input_buffer.assign(buffer_bytes, 0);

    cv::Mat rgb_hwc(model_h, model_w, CV_8UC3, input_buffer.data());
    cv::cvtColor(model_input, rgb_hwc, cv::COLOR_BGR2RGB);
    return true;
}

bool parse_input_hw(const AwnnEngine &engine, const char *tag, int &model_h, int &model_w)
{
    const auto &info = engine.getInfo();
    if (info.input_sizes.empty() || info.input_byte_sizes.empty())
    {
        std::fprintf(stderr, "%s: AwnnEngine has no input tensors\n", tag);
        return false;
    }
    const auto &sizes = info.input_sizes.at(0);
    model_h = static_cast<int>(sizes[1]);
    model_w = static_cast<int>(sizes[2]);
    return true;
}

template <typename Processor>
bool infer_once(
    AwnnEngine &engine,
    const std::vector<std::uint8_t> &input_buffer,
    Processor &processor,
    const char *tag)
{
    if (!engine.run(input_buffer.data(), input_buffer.size()))
    {
        std::fprintf(stderr, "%s: AwnnEngine::run failed\n", tag);
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

    std::printf("yolo_cls_awnn  deploy_percept %s\n", DEPLOY_PERCEPT_VERSION_STRING);
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

    VipLiteRuntime runtime;
    if (!runtime.ok())
    {
        std::fprintf(
            stderr,
            "VipLiteRuntime init failed (set LD_LIBRARY_PATH to VIPLite lib dir)\n");
        return 1;
    }

    AwnnEngine::Params yolo_params;
    yolo_params.model_path = yolo_path;
    AwnnEngine yolo(yolo_params);
    std::printf("yolo: %s", yolo.dump_info().c_str());
    if (!yolo.ok())
    {
        std::fprintf(stderr, "YOLOv8 AwnnEngine init failed\n");
        return 1;
    }

    AwnnEngine::Params cls_params;
    cls_params.model_path = cls_path;
    AwnnEngine cls(cls_params);
    std::printf("cls: %s", cls.dump_info().c_str());
    if (!cls.ok())
    {
        std::fprintf(stderr, "MobileNetV2 AwnnEngine init failed\n");
        return 1;
    }

    int yolo_h = 0;
    int yolo_w = 0;
    int cls_h = 0;
    int cls_w = 0;
    if (!parse_input_hw(yolo, "yolo", yolo_h, yolo_w) ||
        !parse_input_hw(cls, "cls", cls_h, cls_w))
    {
        return 1;
    }
    if (cls.getInfo().output_sizes.empty())
    {
        std::fprintf(stderr, "cls: AwnnEngine has no output tensors\n");
        return 1;
    }
    const int num_classes = static_cast<int>(cls.getInfo().output_sizes.at(0)[0]);

    cv::Mat yolo_bgr;
    std::vector<std::uint8_t> yolo_buf;
    if (!prepare_model_input(
            detect_input,
            yolo_w,
            yolo_h,
            yolo.getInfo().input_byte_sizes.at(0),
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
            cls.getInfo().input_byte_sizes.at(0),
            cls_bgr,
            cls_buf))
    {
        return 1;
    }
    (void)cls_bgr;

    YoloV8DetectPostProcessAwnn::Params yolo_post_params;
    yolo_post_params.model_in_w = yolo_w;
    yolo_post_params.model_in_h = yolo_h;
    YoloV8DetectPostProcessAwnn yolo_post(yolo_post_params);

    ClassificationPostProcessAwnn cls_post({
        .num_classes = num_classes,
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
