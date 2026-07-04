/**
 * Allwinner MobileNetV2 ImageNet 分类 demo（AwnnEngine + ClassificationPostProcessAwnn）
 *
 * 用法（路径相对当前工作目录，或使用绝对路径）：
 *   mobilenetv2_classify_awnn [model.nb] [input.jpg]
 *
 * 默认：./mobilenetv2.nb  ./cat.jpg
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
#include "deploy_percept/core/engine/EngineResultGuard.hpp"
#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"
#include "deploy_percept/utils/fixture_path.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::EngineResultGuard;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::ClassificationPostProcessAwnn;
using deploy_percept::utils::fixture_path;

namespace
{

const fs::path &default_model_path()
{
    static const fs::path path =
        fixture_path("integration/awnn/mobilenetv2_classify/mobilenetv2.nb");
    return path;
}

const fs::path &default_input_path()
{
    static const fs::path path =
        fixture_path("integration/awnn/mobilenetv2_classify/cat.jpg");
    return path;
}

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

} // namespace

int main(int argc, char **argv)
{
    const std::string model_path_str = (argc > 1) ? argv[1] : default_model_path().string();
    const std::string input_path_str = (argc > 2) ? argv[2] : default_input_path().string();
    const char *model_path = model_path_str.c_str();
    const char *input_path = input_path_str.c_str();

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

    EngineResultGuard engine_result_guard(engine);
    if (engine.getResult().tensor_views.empty())
    {
        std::fprintf(stderr, "AwnnEngine output tensors unavailable\n");
        return 1;
    }

    ClassificationPostProcessAwnn processor({});
    if (!processor.run(engine_result_guard.getEngineResult().tensor_views))
    {
        std::fprintf(stderr, "post process failed\n");
        return 1;
    }

    return 0;
}
