#include <gtest/gtest.h>

#include <cstdint>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/core/engine/AwnnEngine.hpp"
#include "deploy_percept/core/engine/EngineResultGuard.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessAwnn.hpp"
#include "deploy_percept/common/types.hpp"
#include "tests/test_common/paths.hpp"
#include "tests/test_common/utils.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::EngineResultGuard;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::Detection;
using deploy_percept::post_process::YoloV8DetectPostProcessAwnn;
using percept::test::fixture_data;

namespace {

constexpr int kBoxTolerancePx = 4;

const std::vector<Detection> kExpectedDetections = {
    MakeDetectResult(16, 0.8922f, 110, 248, 258, 601),
    MakeDetectResult(1, 0.7918f, 100, 148, 474, 467),
    MakeDetectResult(2, 0.6005f, 387, 83, 578, 191),
};

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
        return false;
    }

    cv::Mat resized;
    cv::resize(orig, resized, cv::Size(model_w, model_h));
    input_buffer.assign(buffer_bytes, 0);

    cv::Mat rgb_hwc(model_h, model_w, CV_8UC3, input_buffer.data());
    cv::cvtColor(resized, rgb_hwc, cv::COLOR_BGR2RGB);
    return true;
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

TEST(YoloV8DetectAwnnIntegration, DogDetectionsMatchGolden)
{
    const fs::path model_path = fixture_data("integration/awnn/yolov8_a733.nb");
    const fs::path input_path = fixture_data("integration/awnn/dog.jpg");
    skip_unless_fixture_ready(model_path, input_path);

    VipLiteRuntime runtime;
    if (!runtime.ok())
    {
        GTEST_SKIP() << "VipLiteRuntime init failed (need AWNN NPU + LD_LIBRARY_PATH)";
    }

    AwnnEngine::Params engine_params;
    engine_params.model_path = model_path.string();

    AwnnEngine engine(engine_params);
    if (!engine.is_valid())
    {
        GTEST_SKIP() << "AwnnEngine init failed";
    }

    // yolov8_a733.nb VIP input sizes: [C, H, W, N] = [3, 640, 640, 1]，buffer 为 RGB HWC
    const auto &sizes = engine.getInfo().input_sizes.at(0);
    const int model_c = static_cast<int>(sizes[0]);
    const int model_h = static_cast<int>(sizes[1]);
    const int model_w = static_cast<int>(sizes[2]);
    const std::size_t buffer_bytes = engine.getInfo().input_byte_sizes.at(0);

    std::vector<std::uint8_t> input_buffer;
    ASSERT_TRUE(prepare_model_input(input_path.string(), model_w, model_h, buffer_bytes, input_buffer))
        << "failed to read input: " << input_path;

    ASSERT_TRUE(engine.run(input_buffer.data(), input_buffer.size()));

    EngineResultGuard engine_result_guard(engine);
    ASSERT_FALSE(engine.getResult().tensor_views.empty());
    ASSERT_EQ(engine_result_guard.getEngineResult().tensor_views.size(), 6u);

    YoloV8DetectPostProcessAwnn::Params post_params;
    post_params.model_in_w = model_w;
    post_params.model_in_h = model_h;

    YoloV8DetectPostProcessAwnn processor(post_params);
    ASSERT_TRUE(processor.run(engine_result_guard.getEngineResult().tensor_views));

    expectDetectionsMatchGolden(kExpectedDetections, processor.getResult().detections, kBoxTolerancePx);
}
