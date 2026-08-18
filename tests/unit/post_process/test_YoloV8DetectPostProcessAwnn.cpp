#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "cnpy.h"

#include "deploy_percept/common/types.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessAwnn.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "deploy_percept/utils/vis_draw.hpp"
#include "tests/test_common/fixtures.hpp"
#include "tests/test_common/paths.hpp"
#include "tests/test_common/utils.hpp"

namespace fs = std::filesystem;

using deploy_percept::Detection;
using deploy_percept::TensorView;
using deploy_percept::post_process::YoloV8DetectPostProcessAwnn;
using percept::test::fixture_data;
using percept::test::makeFp32TensorViews;
using percept::test::output_dir;

class YoloV8DetectPostProcessAwnnTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const fs::path npz_path = fixture_data("unit/post_process/yolov8_detect_awnn.npz");
        if (!fs::is_regular_file(npz_path))
        {
            GTEST_SKIP() << "fixture not found (dump on board first): " << npz_path;
        }

        fs::create_directories(output_dir());

        YoloV8DetectPostProcessAwnn::Params params;
        params.model_in_w = 640;
        params.model_in_h = 640;
        processor = std::make_unique<YoloV8DetectPostProcessAwnn>(params);

        model_outputs_npz = cnpy::npz_load(npz_path.string());

        const cv::Mat orig = cv::imread(
            fixture_data("unit/post_process/dog.jpg").string());
        cv::resize(orig, img, cv::Size(params.model_in_w, params.model_in_h));
    }

    std::unique_ptr<YoloV8DetectPostProcessAwnn> processor;
    cnpy::npz_t model_outputs_npz;
    cv::Mat img;
};

TEST_F(YoloV8DetectPostProcessAwnnTest, GoldenRun)
{
    const std::vector<Detection> expected_detections = {
        MakeDetectResult(16, 0.8922f, 110, 248, 258, 601),
        MakeDetectResult(1, 0.7918f, 100, 148, 474, 467),
        MakeDetectResult(2, 0.6005f, 387, 83, 578, 191),
    };

    auto model_outputs = deploy_percept::utils::convertNpzToFloatVectorsByPrefix(
        model_outputs_npz, "output_", 6);

    std::vector<std::vector<float>> storage;
    const std::vector<TensorView> inputs = makeFp32TensorViews(model_outputs, storage);

    ASSERT_TRUE(processor->run(inputs));

    const auto &detection_result = processor->getResult();
    expectDetectionsMatchGolden(expected_detections, detection_result.detections);

    deploy_percept::utils::drawDetectionResults(img, detection_result);
    cv::imwrite((output_dir() / "yolov8_detect_awnn_result.jpg").string(), img);
}
