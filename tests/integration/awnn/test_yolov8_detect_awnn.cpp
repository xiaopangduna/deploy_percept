#include <gtest/gtest.h>

#include <cstdio>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/engine/AwnnImageInput.hpp"
#include "deploy_percept/engine/VipLiteRuntime.hpp"
#include "deploy_percept/engine/AwnnResultGuard.hpp"
#include "deploy_percept/post_process/YoloV8DetectPostProcessAwnn.hpp"
#include "deploy_percept/post_process/types.hpp"
#include "tests/test_common/paths.hpp"
#include "tests/test_common/utils.hpp"
#include "utils/environment.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::AwnnLetterboxMeta;
using deploy_percept::engine::AwnnResultGuard;
using deploy_percept::engine::AwnnRgbInputShape;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::DetectionObject;
using deploy_percept::post_process::YoloV8DetectPostProcessAwnn;
using percept::test::app_data;

namespace {

void print_vip_input_debug(const AwnnEngine &engine)
{
    const auto &info = engine.getInfo();
    std::printf("[yolov8 debug] VIP inputs: count=%zu\n", info.input_sizes.size());
    for (std::size_t i = 0; i < info.input_sizes.size(); ++i)
    {
        const auto &sz = info.input_sizes[i];
        std::printf(
            "  input[%zu]: ndim=%u sizes=[%u,%u,%u,%u] bytes=%u name=%s\n",
            i,
            info.input_num_dims[i],
            sz[0],
            sz[1],
            sz[2],
            sz[3],
            info.input_byte_sizes[i],
            info.input_names[i].c_str());
    }
}

void print_vip_output_debug(const AwnnEngine &engine)
{
    const auto &info = engine.getInfo();
    std::printf("[yolov8 debug] VIP outputs: count=%zu\n", info.output_sizes.size());
    for (std::size_t i = 0; i < info.output_sizes.size(); ++i)
    {
        const auto &sz = info.output_sizes[i];
        const std::size_t elems = info.output_byte_sizes[i] / sizeof(float);
        std::printf(
            "  output[%zu]: ndim=%u sizes=[%u,%u,%u,%u] elems=%zu name=%s",
            i,
            info.output_num_dims[i],
            sz[0],
            sz[1],
            sz[2],
            sz[3],
            elems,
            info.output_names[i].c_str());

        static constexpr int kGridCh = 64;
        static constexpr int kScoreCh = 80;
        for (const int gs : {6400, 1600, 400})
        {
            if (gs > 0 && elems % static_cast<std::size_t>(gs) == 0)
            {
                const std::size_t ch = elems / static_cast<std::size_t>(gs);
                const char *role = (ch == kGridCh) ? "grid" : (ch == kScoreCh) ? "score" : "?";
                std::printf(" gs=%d ch=%zu(%s)", gs, ch, role);
            }
        }
        std::printf("\n");
    }
}

void print_letterbox_debug(
    const AwnnRgbInputShape &shape,
    const cv::Mat &orig_bgr,
    const AwnnLetterboxMeta &meta)
{
    std::printf(
        "[yolov8 debug] orig=%dx%d model=%dx%d scale=%.4f resize=%dx%d pad=[t=%d,b=%d,l=%d,r=%d]\n",
        orig_bgr.cols,
        orig_bgr.rows,
        shape.width,
        shape.height,
        meta.scale,
        meta.resize_w,
        meta.resize_h,
        meta.pad_top,
        meta.pad_bottom,
        meta.pad_left,
        meta.pad_right);
}

void print_detections_debug(const std::vector<DetectionObject> &dets)
{
    std::printf("[yolov8 debug] detections: count=%zu\n", dets.size());
    const std::size_t show = std::min<std::size_t>(dets.size(), 10);
    for (std::size_t i = 0; i < show; ++i)
    {
        const auto &d = dets[i];
        std::printf(
            "  [%zu] cls=%d prob=%.4f box=[%d,%d,%d,%d] (%dx%d)\n",
            i,
            d.cls_id,
            d.prop,
            d.box.left,
            d.box.top,
            d.box.right,
            d.box.bottom,
            d.box.right - d.box.left,
            d.box.bottom - d.box.top);
    }
}

void expect_at_least_one_detection(const std::vector<DetectionObject> &actual)
{
    ASSERT_GE(actual.size(), 1u) << "expected at least one detection on dog.jpg";
    for (const auto &det : actual)
    {
        EXPECT_GE(det.prop, 0.25f);
        EXPECT_LT(det.box.left, det.box.right);
        EXPECT_LT(det.box.top, det.box.bottom);
    }
}

void skip_unless_fixture_ready(const fs::path &model_path, const fs::path &input_path)
{
    if (!fs::is_regular_file(model_path))
    {
        GTEST_SKIP() << "model not found: " << model_path
                     << " (convert yolov8.nb per apps/yolov8_detect_awnn/README.md)";
    }
    if (!fs::is_regular_file(input_path))
    {
        GTEST_SKIP() << "input not found: " << input_path;
    }
}

} // namespace

TEST(YoloV8DetectAwnnIntegration, DogHasDetections)
{
    const fs::path model_path = app_data("yolov8_detect_awnn/yolov8.nb");
    const fs::path input_path = app_data("yolov8_detect_awnn/dog.jpg");
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

    print_vip_input_debug(engine);

    const AwnnRgbInputShape input_shape = deploy_percept::engine::resolveRgbInputShape(engine.getInfo());
    std::printf(
        "[yolov8 debug] resolved input shape: W=%d H=%d C=%d bytes=%zu (HWC interleaved)\n",
        input_shape.width,
        input_shape.height,
        input_shape.channels,
        input_shape.buffer_bytes);

    cv::Mat orig_bgr;
    AwnnLetterboxMeta letterbox_meta{};
    std::vector<std::uint8_t> input_buffer;
    ASSERT_TRUE(deploy_percept::engine::prepareLetterboxRgbInput(
        input_path.string(), input_shape, orig_bgr, input_buffer, &letterbox_meta))
        << "failed to read input: " << input_path;

    print_letterbox_debug(input_shape, orig_bgr, letterbox_meta);

    ASSERT_EQ(input_buffer.size(), input_shape.buffer_bytes);
    ASSERT_TRUE(engine.run(input_buffer.data(), input_buffer.size()));

    print_vip_output_debug(engine);

    AwnnResultGuard engine_result_guard(engine);
    ASSERT_FALSE(engine_result_guard.empty());
    ASSERT_EQ(engine_result_guard.views().size(), 6u);

    YoloV8DetectPostProcessAwnn::Params post_params;
    post_params.model_in_w = input_shape.width;
    post_params.model_in_h = input_shape.height;
    post_params.orig_img_w = orig_bgr.cols;
    post_params.orig_img_h = orig_bgr.rows;

    YoloV8DetectPostProcessAwnn processor(post_params);
    ASSERT_TRUE(processor.run(engine_result_guard.views())) << processor.getResult().message;

    print_detections_debug(processor.getResult().group.detection_objects);
    expect_at_least_one_detection(processor.getResult().group.detection_objects);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);
    return RUN_ALL_TESTS();
}
