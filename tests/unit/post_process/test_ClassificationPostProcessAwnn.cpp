#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "cnpy.h"

#include "deploy_percept/common/types.hpp"
#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "tests/test_common/fixtures.hpp"
#include "tests/test_common/paths.hpp"

namespace fs = std::filesystem;

using deploy_percept::TensorView;
using deploy_percept::post_process::ClassificationPostProcessAwnn;
using percept::test::fixture_data;
using percept::test::makeFp32TensorViews;

class ClassificationPostProcessAwnnTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const fs::path npz_path =
            fixture_data("unit/post_process/mobilenetv2_classify_awnn/model_outputs.npz");
        if (!fs::is_regular_file(npz_path))
        {
            GTEST_SKIP() << "fixture not found (dump on board first): " << npz_path;
        }

        processor = std::make_unique<ClassificationPostProcessAwnn>(ClassificationPostProcessAwnn::Params{
            .num_classes = 1000, // mobilenetv2_a733.nb output_sizes[0][0]（unit 无 Engine，与 fixture 一致）
        });
        model_outputs_npz = cnpy::npz_load(npz_path.string());
    }

    std::unique_ptr<ClassificationPostProcessAwnn> processor;
    cnpy::npz_t model_outputs_npz;
};

TEST_F(ClassificationPostProcessAwnnTest, GoldenRun)
{
    // A733 + mobilenetv2_a733.nb + cat.jpg，与 integration/awnn/test_mobilenetv2_classify_awnn 一致
    const int expected_class_id = 281;
    const float expected_score = 14.8013f; // 板端 dump 原始 logit（非 softmax）

    auto model_outputs = deploy_percept::utils::convertNpzToFloatVectorsByPrefix(
        model_outputs_npz, "output_", 1);

    std::vector<std::vector<float>> storage;
    const std::vector<TensorView> inputs = makeFp32TensorViews(model_outputs, storage);

    ASSERT_TRUE(processor->run(inputs));

    const auto &result = processor->getResult();
    EXPECT_EQ(result.class_id, expected_class_id);
    EXPECT_NEAR(result.score, expected_score, 1e-3f);
}
