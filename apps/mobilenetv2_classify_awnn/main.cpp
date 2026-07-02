/**
 * Allwinner MobileNetV2 ImageNet 分类 demo（AwnnEngine + ClassificationPostProcessAwnn）
 *
 * 用法（路径相对当前工作目录，或使用绝对路径）：
 *   mobilenetv2_classify_awnn [model.nb] [input.jpg] [top_k]
 *
 * 默认：./mobilenetv2.nb  ./cat.jpg  5
 * 性能对比见 benchmarks/awnn/bench_mobilenetv2_classify
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/core/engine/AwnnEngine.hpp"
#include "deploy_percept/core/engine/VipLiteRuntime.hpp"
#include "deploy_percept/core/engine/AwnnResultGuard.hpp"
#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::AwnnResultGuard;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::ClassificationPostProcessAwnn;

namespace
{

constexpr const char *kDefaultModel = "mobilenetv2.nb";
constexpr const char *kDefaultInput = "cat.jpg";
constexpr int kDefaultTopK = 5;

bool prepare_model_input(
    const std::string &input_path,
    const int model_w,
    const int model_h,
    const std::size_t buffer_bytes,
    std::vector<std::uint8_t> &input_buffer)
{
    cv::Mat orig = cv::imread(input_path, cv::IMREAD_COLOR);
    if (orig.empty())
    {
        std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
        return false;
    }

    cv::Mat resized;
    cv::resize(orig, resized, cv::Size(model_w, model_h));
    input_buffer.assign(buffer_bytes, 0);

    cv::Mat rgb_hwc(model_h, model_w, CV_8UC3, input_buffer.data());
    cv::cvtColor(resized, rgb_hwc, cv::COLOR_BGR2RGB);
    return true;
}

void print_top_k(const ClassificationPostProcessAwnn &processor)
{
    const auto &result = processor.getResult();
    std::printf("========== top%d ==========\n", static_cast<int>(result.top_k.size()));
    for (const auto &entry : result.top_k)
    {
        std::printf(
            "class id: %3d, prob: %f, label: %s\n",
            entry.class_id,
            entry.prob,
            entry.label.c_str());
    }
}

} // namespace

int main(int argc, char **argv)
{
    const char *model_path = (argc > 1) ? argv[1] : kDefaultModel;
    const char *input_path = (argc > 2) ? argv[2] : kDefaultInput;
    const int top_k = (argc > 3) ? std::atoi(argv[3]) : kDefaultTopK;

    if (top_k <= 0)
    {
        std::fprintf(stderr, "top_k must be positive\n");
        return 1;
    }

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

    AwnnEngine::Param engine_params;
    engine_params.model_path = model_path;

    AwnnEngine engine(engine_params);
    if (!engine.is_valid())
    {
        std::fprintf(stderr, "AwnnEngine init failed\n");
        return 1;
    }

    // mobilenetv2.nb VIP input sizes: [C, H, W, N] = [3, 224, 224, 1]，buffer 为 RGB HWC
    const auto &sizes = engine.getInfo().input_sizes.at(0);
    const int model_h = static_cast<int>(sizes[1]);
    const int model_w = static_cast<int>(sizes[2]);
    const std::size_t buffer_bytes = engine.getInfo().input_byte_sizes.at(0);

    std::vector<std::uint8_t> input_buffer;
    if (!prepare_model_input(input_path, model_w, model_h, buffer_bytes, input_buffer))
    {
        return 1;
    }

    if (!engine.run(input_buffer.data(), input_buffer.size()))
    {
        std::fprintf(stderr, "AwnnEngine::run failed\n");
        return 1;
    }

    AwnnResultGuard engine_result_guard(engine);
    if (engine_result_guard.empty())
    {
        std::fprintf(stderr, "AwnnEngine output tensors unavailable\n");
        return 1;
    }

    ClassificationPostProcessAwnn::Params post_params;
    post_params.top_k = top_k;

    ClassificationPostProcessAwnn processor(post_params);
    if (!processor.run(engine_result_guard.views()))
    {
        std::fprintf(stderr, "post process failed: %s\n", processor.getResult().message.c_str());
        return 1;
    }

    print_top_k(processor);
    return 0;
}
