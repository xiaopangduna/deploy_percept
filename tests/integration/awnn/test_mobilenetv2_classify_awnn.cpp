#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/core/engine/AwnnEngine.hpp"
#include "deploy_percept/core/engine/VipLiteRuntime.hpp"
#include "deploy_percept/core/engine/AwnnResultGuard.hpp"
#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"
#include "deploy_percept/common/types.hpp"
#include "tests/test_common/paths.hpp"
#include "utils/environment.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::AwnnResultGuard;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::ClassificationPostProcessAwnn;
using percept::test::app_data;

namespace {

// A733 + mobilenetv2.nb + cat.jpg 板端标定（imagenet_10/1.jpg）
const std::vector<int> kExpectedTop5ClassIds = {281, 282, 285, 287, 478};

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

void expect_top_k_class_ids_match(
    const std::vector<int> &expected,
    const std::vector<deploy_percept::ClassificationEntry> &actual)
{
    ASSERT_EQ(expected.size(), actual.size());

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        SCOPED_TRACE("top_k index " + std::to_string(i));
        EXPECT_EQ(expected[i], actual[i].class_id);
    }
}

} // namespace

TEST(MobileNetV2ClassifyAwnnIntegration, CatTop5MatchGolden)
{
    const fs::path model_path = app_data("mobilenetv2_classify_awnn/mobilenetv2.nb");
    const fs::path input_path = app_data("mobilenetv2_classify_awnn/cat.jpg");
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
    const int model_h = static_cast<int>(sizes[1]);
    const int model_w = static_cast<int>(sizes[2]);
    const std::size_t buffer_bytes = engine.getInfo().input_byte_sizes.at(0);

    std::vector<std::uint8_t> input_buffer;
    ASSERT_TRUE(prepare_model_input(input_path.string(), model_w, model_h, buffer_bytes, input_buffer))
        << "failed to read input: " << input_path;

    ASSERT_TRUE(engine.run(input_buffer.data(), input_buffer.size()));

    AwnnResultGuard engine_result_guard(engine);
    ASSERT_FALSE(engine_result_guard.empty());
    ASSERT_GE(engine_result_guard.views().size(), 1u);

    ClassificationPostProcessAwnn::Params post_params;
    post_params.top_k = 5;

    ClassificationPostProcessAwnn processor(post_params);
    ASSERT_TRUE(processor.run(engine_result_guard.views())) << processor.getResult().message;

    const auto &result = processor.getResult();
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.top_k.size(), kExpectedTop5ClassIds.size());

    expect_top_k_class_ids_match(kExpectedTop5ClassIds, result.top_k);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);
    return RUN_ALL_TESTS();
}
