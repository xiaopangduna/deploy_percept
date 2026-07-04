#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "cnpy.h"

#include "deploy_percept/common/types.hpp"
#include "deploy_percept/core/post_process/YoloV5DetectPostProcessAwnn.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "deploy_percept/utils/vis_draw.hpp"
#include "tests/test_common/fixtures.hpp"
#include "tests/test_common/paths.hpp"
#include "tests/test_common/utils.hpp"

namespace fs = std::filesystem;

using deploy_percept::Detection;
using deploy_percept::TensorView;
using deploy_percept::post_process::YoloV5DetectPostProcessAwnn;
using percept::test::fixture_data;
using percept::test::makeFp32TensorViews;
using percept::test::output_dir;

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

        fs::create_directories(output_dir());

        YoloV5DetectPostProcessAwnn::Params params;
        params.model_in_w = 640;
        params.model_in_h = 640;
        processor = std::make_unique<YoloV5DetectPostProcessAwnn>(params);

        model_outputs_npz = cnpy::npz_load(npz_path.string());

        const cv::Mat orig = cv::imread(
            fixture_data("unit/post_process/yolov5_detect_awnn/dog.jpg").string());
        cv::resize(orig, img, cv::Size(params.model_in_w, params.model_in_h));
    }

    std::unique_ptr<YoloV5DetectPostProcessAwnn> processor;
    cnpy::npz_t model_outputs_npz;
    cv::Mat img;
};

TEST_F(YoloV5DetectPostProcessAwnnTest, GoldenRun)
{
    const std::vector<Detection> expected_detections = {
        MakeDetectResult(16, 0.8489f, 111, 236, 256, 611),
        MakeDetectResult(7, 0.6496f, 388, 82, 575, 193),
        MakeDetectResult(1, 0.4271f, 91, 144, 464, 468),
    };

    auto model_outputs = deploy_percept::utils::convertNpzToFloatVectorsByPrefix(
        model_outputs_npz, "output_", 3);

    std::vector<std::vector<float>> storage;
    const std::vector<TensorView> inputs = makeFp32TensorViews(model_outputs, storage);

    ASSERT_TRUE(processor->run(inputs));

    const auto &detection_result = processor->getResult();
    expectDetectionsMatchGolden(expected_detections, detection_result.detections);

    deploy_percept::utils::drawDetectionResults(img, detection_result);
    cv::imwrite((output_dir() / "yolov5_detect_awnn_result.jpg").string(), img);
}
