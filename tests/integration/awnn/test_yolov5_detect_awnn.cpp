#include <gtest/gtest.h>

#include <cstdint>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/engine/VipLiteRuntime.hpp"
#include "deploy_percept/engine/AwnnResultGuard.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"
#include "deploy_percept/post_process/types.hpp"
#include "tests/test_common/paths.hpp"
#include "tests/test_common/utils.hpp"
#include "utils/environment.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::AwnnResultGuard;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::DetectionObject;
using deploy_percept::post_process::YoloV5DetectPostProcessAwnn;
using percept::test::app_data;

namespace {

constexpr int kBoxTolerancePx = 2;

// A733 + yolov5.nb + dog.jpg 板端标定
const std::vector<DetectionObject> kExpectedDetections = {
    MakeDetectResult(16, "class_16", 0.3111f, 111, 236, 256, 611),
    MakeDetectResult(7, "class_7", 0.7592f, 388, 82, 575, 193),
    MakeDetectResult(1, "class_1", 0.4049f, 91, 144, 464, 468),
};

/**
 * 将 BGR 图像 pack 为 VIP input buffer 字节序。
 * VIP input_sizes 顺序为 [W, H, C, N]（vip_lite.h）；buffer 线性布局 w 最快变。
 * batch=1 时与 ai-sdk yolov5 demo 的 pack 方式一致。
 */
std::vector<std::uint8_t> pack_rgb_for_vip_input(
    const cv::Mat &bgr,
    const int width,
    const int height,
    const int channels)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(width * height * channels));
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            const cv::Vec3b pix = rgb.at<cv::Vec3b>(h, w);
            for (int c = 0; c < channels; ++c)
            {
                buffer[c * height * width + h * width + w] = pix[c];
            }
        }
    }
    return buffer;
}

bool prepare_model_input(
    const std::string &input_path,
    const int model_w,
    const int model_h,
    const int model_c,
    std::vector<std::uint8_t> &input_buffer)
{
    cv::Mat orig = cv::imread(input_path, cv::IMREAD_COLOR);
    if (orig.empty())
    {
        return false;
    }

    cv::Mat resized;
    cv::resize(orig, resized, cv::Size(model_w, model_h));
    input_buffer = pack_rgb_for_vip_input(resized, model_w, model_h, model_c);
    return true;
}

void expect_class_and_box_match(
    const std::vector<DetectionObject> &expected,
    const std::vector<DetectionObject> &actual)
{
    ASSERT_EQ(expected.size(), actual.size());

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const auto &exp = expected[i];
        const auto &act = actual[i];

        SCOPED_TRACE("detection index " + std::to_string(i));

        EXPECT_EQ(exp.cls_id, act.cls_id);

        EXPECT_LE(std::abs(exp.box.left - act.box.left), kBoxTolerancePx) << "box.left";
        EXPECT_LE(std::abs(exp.box.top - act.box.top), kBoxTolerancePx) << "box.top";
        EXPECT_LE(std::abs(exp.box.right - act.box.right), kBoxTolerancePx) << "box.right";
        EXPECT_LE(std::abs(exp.box.bottom - act.box.bottom), kBoxTolerancePx) << "box.bottom";
    }
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

TEST(YoloV5DetectAwnnIntegration, DogDetectionsMatchGolden)
{
    const fs::path model_path = app_data("yolov5_detect_awnn/yolov5.nb");
    const fs::path input_path = app_data("yolov5_detect_awnn/dog.jpg");
    skip_unless_fixture_ready(model_path, input_path);

    VipLiteRuntime runtime;
    if (!runtime.ok())
    {
        GTEST_SKIP() << "VipLiteRuntime init failed (need AWNN NPU + LD_LIBRARY_PATH)";
    }

    AwnnEngine::Param engine_params;
    engine_params.model_path = model_path.string();

    AwnnEngine engine(engine_params);
    if (!engine.is_valid())
    {
        GTEST_SKIP() << "AwnnEngine init failed";
    }

    const auto &sizes = engine.getInfo().input_sizes.at(0);
    const int model_w = static_cast<int>(sizes[0]);
    const int model_h = static_cast<int>(sizes[1]);
    const int model_c = static_cast<int>(sizes[2]);

    std::vector<std::uint8_t> input_buffer;
    ASSERT_TRUE(prepare_model_input(input_path.string(), model_w, model_h, model_c, input_buffer))
        << "failed to read input: " << input_path;

    ASSERT_TRUE(engine.run(input_buffer.data(), input_buffer.size()));

    AwnnResultGuard engine_result_guard(engine);
    ASSERT_FALSE(engine_result_guard.empty());

    YoloV5DetectPostProcessAwnn::Params post_params;
    post_params.model_in_w = model_w;
    post_params.model_in_h = model_h;

    YoloV5DetectPostProcessAwnn processor(post_params);
    ASSERT_TRUE(processor.run(engine_result_guard.views())) << processor.getResult().message;

    expect_class_and_box_match(kExpectedDetections, processor.getResult().group.detection_objects);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);
    return RUN_ALL_TESTS();
}
