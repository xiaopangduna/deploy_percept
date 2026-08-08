#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "deploy_percept/common/types.hpp"
#include "deploy_percept/core/post_process/ClassificationPostProcessSigmastar.hpp"

using deploy_percept::TensorDtype;
using deploy_percept::TensorView;
using deploy_percept::post_process::ClassificationPostProcessSigmastar;

TEST(ClassificationPostProcessSigmastarTest, Int16SoftmaxTop1)
{
    // logits = [0, 1, 2] * scalar(0.5) = [0, 0.5, 1.0] → Softmax → argmax=2
    const std::vector<std::int16_t> data = {0, 1, 2};
    const float scalar = 0.5f;

    ClassificationPostProcessSigmastar processor({
        .num_classes = 3,
        .output_scalar = scalar,
    });

    TensorView view{};
    view.data = data.data();
    view.byte_size = data.size() * sizeof(std::int16_t);
    view.dtype = TensorDtype::INT16;

    ASSERT_TRUE(processor.run({view}));

    const auto &result = processor.getResult();
    EXPECT_EQ(result.class_id, 2);

    const float e0 = std::exp(0.f);
    const float e1 = std::exp(0.5f);
    const float e2 = std::exp(1.f);
    const float expected = e2 / (e0 + e1 + e2);
    EXPECT_NEAR(result.score, expected, 1e-5f);
}

TEST(ClassificationPostProcessSigmastarTest, Fp32SoftmaxTop1)
{
    const std::vector<float> data = {1.f, 3.f, 2.f};

    ClassificationPostProcessSigmastar processor({
        .num_classes = 3,
        .output_scalar = 1.f,
    });

    TensorView view{};
    view.data = data.data();
    view.byte_size = data.size() * sizeof(float);
    view.dtype = TensorDtype::FP32;

    ASSERT_TRUE(processor.run({view}));

    const auto &result = processor.getResult();
    EXPECT_EQ(result.class_id, 1);

    const float e0 = std::exp(1.f);
    const float e1 = std::exp(3.f);
    const float e2 = std::exp(2.f);
    const float expected = e1 / (e0 + e1 + e2);
    EXPECT_NEAR(result.score, expected, 1e-5f);
}

TEST(ClassificationPostProcessSigmastarTest, RejectsShortBuffer)
{
    const std::vector<std::int16_t> data = {1, 2};

    ClassificationPostProcessSigmastar processor({
        .num_classes = 3,
        .output_scalar = 1.f,
    });

    TensorView view{};
    view.data = data.data();
    view.byte_size = data.size() * sizeof(std::int16_t);
    view.dtype = TensorDtype::INT16;

    EXPECT_FALSE(processor.run({view}));
}
