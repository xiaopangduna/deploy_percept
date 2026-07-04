#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "cnpy.h"

#include "deploy_percept/common/types.hpp"
#include "deploy_percept/core/post_process/YoloV5DetectPostProcessRknn.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "deploy_percept/utils/vis_draw.hpp"
#include "tests/test_common/fixtures.hpp"
#include "tests/test_common/paths.hpp"
#include "tests/test_common/utils.hpp"

namespace fs = std::filesystem;

using deploy_percept::Detection;
using deploy_percept::post_process::YoloV5DetectPostProcessRknn;
using percept::test::fixture_data;
using percept::test::makeInt8PtrInputs;
using percept::test::output_dir;

class YoloV5DetectPostProcessRknnTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        fs::create_directories(output_dir());

        processor = std::make_unique<YoloV5DetectPostProcessRknn>(YoloV5DetectPostProcessRknn::Params{});

        model_outputs_npz = cnpy::npz_load(
            fixture_data("unit/post_process/yolov5_detect/model_outputs.npz").string());

        img = cv::imread(fixture_data("unit/post_process/yolov5_detect/bus.jpg").string());
    }

    std::unique_ptr<YoloV5DetectPostProcessRknn> processor;
    cnpy::npz_t model_outputs_npz;
    cv::Mat img;
};

TEST_F(YoloV5DetectPostProcessRknnTest, GoldenRun)
{
    const std::vector<Detection> expected_detections = {
        MakeDetectResult(0, 0.8797f, 209, 243, 286, 510),
        MakeDetectResult(0, 0.8706f, 479, 238, 560, 526),
        MakeDetectResult(0, 0.8398f, 109, 238, 231, 534),
        MakeDetectResult(5, 0.6920f, 91, 129, 555, 464),
        MakeDetectResult(0, 0.3010f, 79, 353, 121, 517),
    };

    auto model_outputs = deploy_percept::utils::convertNpzToInt8VectorsByPrefix(
        model_outputs_npz, "output_", 3);

    const std::vector<int8_t *> inputs = makeInt8PtrInputs(model_outputs);

    std::vector<int32_t> qnt_zps = {-128, -128, -128};
    std::vector<float> qnt_scales = {0.00392157f, 0.00392157f, 0.00392157f};

    ASSERT_TRUE(processor->run(inputs, 640, 640, qnt_zps, qnt_scales));

    const auto &detection_result = processor->getResult();
    expectDetectionsMatchGolden(expected_detections, detection_result.detections);

    deploy_percept::utils::drawDetectionResults(img, detection_result);
    cv::imwrite((output_dir() / "yolov5_detect_rknn_result.jpg").string(), img);
}
