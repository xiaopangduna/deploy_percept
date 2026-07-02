#include <gtest/gtest.h>

#include <vector>

#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"
#include "deploy_percept/common/types.hpp"
#include "deploy_percept/data/imagenet_labels.hpp"

using deploy_percept::post_process::ClassificationPostProcessAwnn;
using deploy_percept::post_process::TensorDtype;
using deploy_percept::post_process::TensorView;

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

TEST(ClassificationPostProcessAwnnTest, TopKReturnsHighestClasses)
{
    std::vector<float> logits = make_logits_with_peaks({42, 7, 281, 999, 123, 456}, 9.5f);
    logits[static_cast<std::size_t>(42)] = 10.f;
    logits[static_cast<std::size_t>(7)] = 9.9f;
    logits[static_cast<std::size_t>(281)] = 9.8f;
    logits[static_cast<std::size_t>(999)] = 9.7f;
    logits[static_cast<std::size_t>(123)] = 9.6f;

    ClassificationPostProcessAwnn::Params params;
    params.top_k = 5;
    ClassificationPostProcessAwnn processor(params);

    const TensorView view{logits.data(), logits.size() * sizeof(float), TensorDtype::FP32};
    ASSERT_TRUE(processor.run({view}));

    const auto &result = processor.getResult();
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.top_k.size(), 5u);
    EXPECT_EQ(result.top_k[0].class_id, 42);
    EXPECT_EQ(result.top_k[1].class_id, 7);
    EXPECT_EQ(result.top_k[2].class_id, 281);
    EXPECT_EQ(result.top_k[3].class_id, 999);
    EXPECT_EQ(result.top_k[4].class_id, 123);
    EXPECT_STREQ(result.top_k[2].label.c_str(), deploy_percept::data::imagenet_label(281));
}

TEST(ClassificationPostProcessAwnnTest, RejectsEmptyInput)
{
    ClassificationPostProcessAwnn processor({});
    EXPECT_FALSE(processor.run({}));
    EXPECT_FALSE(processor.getResult().success);
}

TEST(ClassificationPostProcessAwnnTest, RejectsUndersizedOutput)
{
    std::vector<float> logits(10, 1.f);
    const TensorView view{logits.data(), logits.size() * sizeof(float), TensorDtype::FP32};

    ClassificationPostProcessAwnn processor({});
    EXPECT_FALSE(processor.run({view}));
}
