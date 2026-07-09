/**
 * Allwinner YOLOv8 检测 demo（AwnnEngine + YoloV8DetectPostProcessAwnn）
 *
 * 用法（路径相对当前工作目录，或使用绝对路径）：
 *   yolov8_detect_awnn [model.nb] [input.jpg] [output.jpg] [model_outputs.npz]
 *
 * 默认：./yolov8.nb  ./dog.jpg  ./yolov8_detect_awnn_out.jpg
 * 性能对比见 tests/benchmarks/awnn/bench_yolov8
 */
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/core/engine/AwnnEngine.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessAwnn.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "deploy_percept/utils/vis_draw.hpp"
#include "deploy_percept/data/class_labels.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::YoloV8DetectPostProcessAwnn;
using deploy_percept::data::kCoco80Labels;
using deploy_percept::utils::drawDetectionResults;
using deploy_percept::utils::convertTensorViewsToNpz;
using deploy_percept::utils::save_npz;

namespace
{

constexpr const char *kDefaultModel = "yolov8.nb";
constexpr const char *kDefaultInput = "dog.jpg";
constexpr const char *kDefaultOutput = "yolov8_detect_awnn_out.jpg";

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

} // namespace

int main(int argc, char **argv)
{
    const std::string model_path_str = (argc > 1) ? argv[1] : kDefaultModel;
    const std::string input_path_str = (argc > 2) ? argv[2] : kDefaultInput;
    const char *model_path = model_path_str.c_str();
    const char *input_path = input_path_str.c_str();
    const char *output_path = (argc > 3) ? argv[3] : kDefaultOutput;
    const char *npz_path = (argc > 4) ? argv[4] : nullptr;

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

    VipLiteRuntime runtime;
    if (!runtime.ok())
    {
        std::fprintf(
            stderr,
            "VipLiteRuntime init failed (set LD_LIBRARY_PATH to VIPLite lib dir)\n");
        return 1;
    }

    AwnnEngine::Params engine_params;
    engine_params.model_path = model_path;
    // 默认：is_copy_host=true，run() 内输出拷至 host，post 前可直接 getResult()
    // 零拷贝 mapped（非默认）：取消下行注释，并改用下方 EngineResultGuard 取 tensor_views
    // （需 #include "deploy_percept/core/engine/EngineResultGuard.hpp"）
    // engine_params.is_copy_host = false;

    AwnnEngine engine(engine_params);
    if (!engine.is_valid())
    {
        std::fprintf(stderr, "AwnnEngine init failed\n");
        return 1;
    }

    // yolov8.nb VIP input sizes: [C, H, W, N] = [3, 640, 640, 1]，buffer 为 RGB HWC
    const auto &sizes = engine.getInfo().input_sizes.at(0);
    const int model_h = static_cast<int>(sizes[1]);
    const int model_w = static_cast<int>(sizes[2]);
    const std::size_t buffer_bytes = engine.getInfo().input_byte_sizes.at(0);

    cv::Mat model_input;
    std::vector<std::uint8_t> input_buffer;
    if (!prepare_model_input(input_path, model_w, model_h, buffer_bytes, model_input, input_buffer))
    {
        return 1;
    }

    if (!engine.run(input_buffer.data(), input_buffer.size()))
    {
        std::fprintf(stderr, "AwnnEngine::run failed\n");
        return 1;
    }

    // 默认（is_copy_host=true）：输出已在 host，直接读取
    const auto &tensor_views = engine.getResult().tensor_views;
    // 零拷贝 mapped（is_copy_host=false 时改用）：
    // deploy_percept::engine::EngineResultGuard engine_result_guard(engine);
    // const auto &tensor_views = engine_result_guard.getEngineResult().tensor_views;
    if (tensor_views.empty())
    {
        std::fprintf(stderr, "AwnnEngine output tensors unavailable\n");
        return 1;
    }

    std::error_code ec;

    if (npz_path != nullptr)
    {
        try
        {
            const cnpy::npz_t model_outputs_npz = convertTensorViewsToNpz(tensor_views);
            fs::create_directories(fs::path(npz_path).parent_path(), ec);
            if (!save_npz(npz_path, model_outputs_npz))
            {
                std::fprintf(stderr, "failed to save model outputs to %s\n", npz_path);
                return 1;
            }
            std::printf("saved model outputs to %s\n", npz_path);
        }
        catch (const std::exception &ex)
        {
            std::fprintf(stderr, "failed to dump npz: %s\n", ex.what());
            return 1;
        }
    }

    YoloV8DetectPostProcessAwnn::Params post_params;
    post_params.model_in_w = model_w;
    post_params.model_in_h = model_h;

    YoloV8DetectPostProcessAwnn processor(post_params);
    if (!processor.run(tensor_views))
    {
        std::fprintf(stderr, "post process failed\n");
        return 1;
    }

    cv::Mat result_img = model_input.clone();
    drawDetectionResults(result_img, processor.getResult(), &kCoco80Labels);

    fs::create_directories(fs::path(output_path).parent_path(), ec);
    if (!cv::imwrite(output_path, result_img))
    {
        std::fprintf(stderr, "failed to save output to %s\n", output_path);
        return 1;
    }

    std::printf("detection num: %zu\n", processor.getResult().detections.size());
    std::printf("saved detect result to %s\n", output_path);
    return 0;
}
