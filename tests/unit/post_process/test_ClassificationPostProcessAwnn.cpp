#include <gtest/gtest.h>

#include <vector>

#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"
#include "deploy_percept/common/types.hpp"

using deploy_percept::post_process::ClassificationPostProcessAwnn;
using deploy_percept::TensorDtype;
using deploy_percept::TensorView;

namespace {

std::vector<float> make_logits_with_peaks(const std::vector<int> &peak_indices, const float peak_value)
{
    std::vector<float> logits(1000, 0.f);
    for (int idx : peak_indices)
    {
        logits[static_cast<std::size_t>(idx)] = peak_value;
    }
    return logits;
}

} // namespace

TEST(ClassificationPostProcessAwnnTest, ArgmaxReturnsHighestClass)
{
    std::vector<float> logits = make_logits_with_peaks({42, 7, 281}, 9.5f);
    logits[static_cast<std::size_t>(42)] = 10.f;

    ClassificationPostProcessAwnn processor({});
    const TensorView view{logits.data(), logits.size() * sizeof(float), TensorDtype::FP32};
    ASSERT_TRUE(processor.run({view}));

    const auto &result = processor.getResult();
    EXPECT_EQ(result.class_id, 42);
    EXPECT_FLOAT_EQ(result.score, 10.f);
}

TEST(ClassificationPostProcessAwnnTest, RejectsEmptyInput)
{
    ClassificationPostProcessAwnn processor({});
    EXPECT_FALSE(processor.run({}));
}

TEST(ClassificationPostProcessAwnnTest, RejectsUndersizedOutput)
{
    std::vector<float> logits(10, 1.f);
    const TensorView view{logits.data(), logits.size() * sizeof(float), TensorDtype::FP32};

    ClassificationPostProcessAwnn processor({});
    EXPECT_FALSE(processor.run({view}));
}
