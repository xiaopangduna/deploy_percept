#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "cnpy.h"

#include "deploy_percept/common/types.hpp"
#include "deploy_percept/core/post_process/YoloV5DetectPostProcessAwnn.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "tests/test_common/paths.hpp"
#include "tests/test_common/utils.hpp"

namespace fs = std::filesystem;

using deploy_percept::Detection;
using deploy_percept::TensorDtype;
using deploy_percept::TensorView;
using deploy_percept::post_process::YoloV5DetectPostProcessAwnn;
using percept::test::fixture_data;

namespace {

const std::vector<Detection> kExpectedDetections = {
    MakeDetectResult(16, 0.8489f, 111, 236, 256, 611),
    MakeDetectResult(7, 0.6496f, 388, 82, 575, 193),
    MakeDetectResult(1, 0.4271f, 91, 144, 464, 468),
};

std::vector<TensorView> make_fp32_tensor_views(
    const std::vector<std::vector<float>> &buffers,
    std::vector<std::vector<float>> &storage)
{
    storage = buffers;
    std::vector<TensorView> views;
    views.reserve(storage.size());
    for (const auto &buf : storage)
    {
        views.push_back({buf.data(), buf.size() * sizeof(float), TensorDtype::FP32});
    }
    return views;
}

} // namespace

class YoloV5DetectPostProcessAwnnTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const fs::path npz_path = fixture_data("unit/post_process/yolov5_detect_awnn/model_outputs.npz");
        if (!fs::is_regular_file(npz_path))
        {
            GTEST_SKIP() << "fixture not found (dump on board first): " << npz_path;
        }

        YoloV5DetectPostProcessAwnn::Params params;
        params.model_in_w = 640;
        params.model_in_h = 640;
        processor = std::make_unique<YoloV5DetectPostProcessAwnn>(params);

        model_outputs_npz = cnpy::npz_load(npz_path.string());
    }

    std::unique_ptr<YoloV5DetectPostProcessAwnn> processor;
    cnpy::npz_t model_outputs_npz;
};

TEST_F(YoloV5DetectPostProcessAwnnTest, GoldenRun)
{
    auto model_outputs = deploy_percept::utils::convertNpzToFloatVectorsByPrefix(
        model_outputs_npz, "output_", 3);
    ASSERT_EQ(model_outputs.size(), 3u);

    std::vector<std::vector<float>> storage;
    const std::vector<TensorView> inputs = make_fp32_tensor_views(model_outputs, storage);

    ASSERT_TRUE(processor->run(inputs));
    EXPECT_TRUE(isDetectionVectorEqualWithinTolerance(
        kExpectedDetections, processor->getResult().detections));
}
