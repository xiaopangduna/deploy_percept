#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "deploy_percept/common/types.hpp"
#include "deploy_percept/core/post_process/YoloV8SegPostProcessSigmastar.hpp"

using deploy_percept::TensorDtype;
using deploy_percept::TensorView;
using deploy_percept::post_process::YoloV8SegPostProcessSigmastar;

TEST(YoloV8SegPostProcessSigmastarTest, RejectsWrongOutputCount)
{
    YoloV8SegPostProcessSigmastar processor({
        .model_in_h = 640,
        .model_in_w = 640,
        .output_shapes = {{1, 116, 8400}},
        .output_scalars = {1.f},
    });

    TensorView view{};
    view.data = nullptr;
    view.byte_size = 0;
    view.dtype = TensorDtype::FP32;

    EXPECT_FALSE(processor.run({view}));
}

TEST(YoloV8SegPostProcessSigmastarTest, FusedFp32OneBoxWithMask)
{
    // Minimal: 1 box, C=4+2+2=8 (2 classes, 2 proto), N=1
    // Proto: H=2 W=2 C=2 HWC
    constexpr int kNumClasses = 2;
    constexpr int kMaskC = 2;
    constexpr int kChannels = 4 + kNumClasses + kMaskC;
    constexpr int kNumBoxes = 1;
    constexpr int kMaskH = 2;
    constexpr int kMaskW = 2;

    std::vector<float> det(static_cast<std::size_t>(kChannels) * kNumBoxes, 0.f);
    // cx,cy,w,h in net pixels (model 4x4)
    det[0] = 2.f; // cx
    det[1] = 2.f; // cy
    det[2] = 2.f; // w
    det[3] = 2.f; // h
    det[4] = 0.1f; // class0
    det[5] = 0.9f; // class1
    det[6] = 1.f;  // m0
    det[7] = 1.f;  // m1

    // proto HWC: for each pixel, channels [1,1] → sum = 1*1 + 1*1 = 2 > logit(0.5)=0
    std::vector<float> proto(static_cast<std::size_t>(kMaskH * kMaskW * kMaskC), 1.f);

    YoloV8SegPostProcessSigmastar processor({
        .model_in_h = 4,
        .model_in_w = 4,
        .conf_threshold = 0.5f,
        .nms_threshold = 0.5f,
        .mask_threshold = 0.5f,
        .num_classes = kNumClasses,
        .output_shapes =
            {
                {1, static_cast<std::uint32_t>(kChannels), kNumBoxes},
                {1, kMaskH, kMaskW, static_cast<std::uint32_t>(kMaskC)},
            },
        .output_scalars = {1.f, 1.f},
    });

    TensorView det_view{};
    det_view.data = det.data();
    det_view.byte_size = det.size() * sizeof(float);
    det_view.dtype = TensorDtype::FP32;

    TensorView proto_view{};
    proto_view.data = proto.data();
    proto_view.byte_size = proto.size() * sizeof(float);
    proto_view.dtype = TensorDtype::FP32;

    ASSERT_TRUE(processor.run({det_view, proto_view}));
    const auto &result = processor.getResult();
    ASSERT_EQ(result.detections.size(), 1u);
    EXPECT_EQ(result.detections[0].class_id, 1);
    ASSERT_EQ(result.label_mask.size(), 16u);

    std::size_t nnz = 0;
    for (std::uint8_t v : result.label_mask)
    {
        if (v != 0)
        {
            ++nnz;
            EXPECT_EQ(v, 2); // class_id + 1
        }
    }
    EXPECT_GT(nnz, 0u);
}
