#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "deploy_percept/common/types.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessSigmastar.hpp"

using deploy_percept::TensorDtype;
using deploy_percept::TensorView;
using deploy_percept::post_process::YoloV8DetectPostProcessSigmastar;

namespace
{

TensorView makeView(const void *data, std::size_t bytes, TensorDtype dtype)
{
    TensorView view{};
    view.data = data;
    view.byte_size = bytes;
    view.dtype = dtype;
    return view;
}

} // namespace

TEST(YoloV8DetectPostProcessSigmastarTest, PureIpuFp32CoordLast)
{
    // pcupid with_filter: boxes [1, N, 4] + FP32 score_number / score_index
    constexpr int kNumAll = 2;
    const std::vector<float> boxes = {
        10.f, 20.f, 30.f, 40.f,
        50.f, 60.f, 90.f, 100.f,
    };
    const std::vector<float> classes = {1.f, 16.f};
    const std::vector<float> scores = {0.91f, 0.82f};
    const std::vector<float> score_index = {1.f, 0.f};
    const std::vector<float> score_number = {2.f};

    YoloV8DetectPostProcessSigmastar processor({
        .model_in_h = 640,
        .model_in_w = 640,
        .conf_threshold = 0.5f,
        .nms_threshold = 0.5f,
        .output_shapes =
            {
                {1, kNumAll, 4},
                {1, kNumAll, 1},
                {1, kNumAll},
                {1, kNumAll},
                {1},
            },
        .output_scalars = {1.f, 1.f, 1.f, 1.f, 1.f},
    });

    ASSERT_TRUE(processor.run({
        makeView(boxes.data(), boxes.size() * sizeof(float), TensorDtype::FP32),
        makeView(classes.data(), classes.size() * sizeof(float), TensorDtype::FP32),
        makeView(scores.data(), scores.size() * sizeof(float), TensorDtype::FP32),
        makeView(score_index.data(), score_index.size() * sizeof(float), TensorDtype::FP32),
        makeView(score_number.data(), score_number.size() * sizeof(float), TensorDtype::FP32),
    }));

    const auto &dets = processor.getResult().detections;
    ASSERT_EQ(dets.size(), 2u);
    EXPECT_EQ(dets[0].class_id, 1);
    EXPECT_NEAR(dets[0].score, 0.91f, 1e-5f);
    EXPECT_EQ(dets[0].box.left, 10);
    EXPECT_EQ(dets[0].box.top, 20);
    EXPECT_EQ(dets[0].box.right, 30);
    EXPECT_EQ(dets[0].box.bottom, 40);
    EXPECT_EQ(dets[1].class_id, 16);
    EXPECT_NEAR(dets[1].score, 0.82f, 1e-5f);
}

TEST(YoloV8DetectPostProcessSigmastarTest, PureIpuInt16ChannelFirst)
{
    constexpr int kNumAll = 2;
    // layout [4, N]: x0,x1, y0,y1, x2_0,x2_1, y2_0,y2_1
    const std::vector<float> boxes = {10.f, 50.f, 20.f, 60.f, 30.f, 90.f, 40.f, 100.f};
    const std::vector<std::int16_t> classes = {1, 16};
    const std::vector<float> scores = {0.91f, 0.82f};
    const std::vector<std::int16_t> score_index = {0, 1};
    const std::vector<std::int16_t> score_number = {1};

    YoloV8DetectPostProcessSigmastar processor({
        .model_in_h = 640,
        .model_in_w = 640,
        .conf_threshold = 0.5f,
        .nms_threshold = 0.5f,
        .output_shapes =
            {
                {1, 4, kNumAll},
                {1, kNumAll, 1},
                {1, kNumAll},
                {1, kNumAll},
                {1},
            },
        .output_scalars = {1.f, 1.f, 1.f, 1.f, 1.f},
    });

    ASSERT_TRUE(processor.run({
        makeView(boxes.data(), boxes.size() * sizeof(float), TensorDtype::FP32),
        makeView(classes.data(), classes.size() * sizeof(std::int16_t), TensorDtype::INT16),
        makeView(scores.data(), scores.size() * sizeof(float), TensorDtype::FP32),
        makeView(score_index.data(), score_index.size() * sizeof(std::int16_t), TensorDtype::INT16),
        makeView(score_number.data(), score_number.size() * sizeof(std::int16_t), TensorDtype::INT16),
    }));

    const auto &dets = processor.getResult().detections;
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].class_id, 1);
    EXPECT_EQ(dets[0].box.left, 10);
    EXPECT_EQ(dets[0].box.top, 20);
    EXPECT_EQ(dets[0].box.right, 30);
    EXPECT_EQ(dets[0].box.bottom, 40);
}

TEST(YoloV8DetectPostProcessSigmastarTest, FusedFp32OneBox)
{
    const std::vector<float> fused = {2.f, 3.f, 4.f, 2.f, 0.1f, 0.9f};

    YoloV8DetectPostProcessSigmastar processor({
        .model_in_h = 8,
        .model_in_w = 8,
        .conf_threshold = 0.5f,
        .nms_threshold = 0.5f,
        .output_shapes = {{1, 6, 1}},
        .output_scalars = {1.f},
    });

    ASSERT_TRUE(processor.run(
        {makeView(fused.data(), fused.size() * sizeof(float), TensorDtype::FP32)}));

    const auto &dets = processor.getResult().detections;
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].class_id, 1);
    EXPECT_NEAR(dets[0].score, 0.9f, 1e-5f);
}
