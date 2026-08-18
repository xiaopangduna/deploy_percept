#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "cnpy.h"

#include "deploy_percept/core/post_process/YoloV8SegPostProcessRknn.hpp"
#include "deploy_percept/utils/io.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "deploy_percept/utils/vis_draw.hpp"
#include "tests/test_common/fixtures.hpp"
#include "tests/test_common/paths.hpp"
#include "tests/test_common/utils.hpp"

namespace fs = std::filesystem;

using deploy_percept::Detection;
using deploy_percept::post_process::YoloV8SegPostProcessRknn;
using deploy_percept::utils::loadUint8VectorFromBinFile;
using percept::test::fixture_data;
using percept::test::makeInt8PtrInputs;
using percept::test::output_dir;

class YoloV8SegPostProcessRknnTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        fs::create_directories(output_dir());

        processor = std::make_unique<YoloV8SegPostProcessRknn>(YoloV8SegPostProcessRknn::Params{});

        model_outputs_npz = cnpy::npz_load(
            fixture_data("unit/post_process/yolov8_seg_rknn.npz").string());

        path_seg_result = fixture_data("unit/post_process/yolov8_seg_rknn_mask.bin");
        img = cv::imread(fixture_data("unit/post_process/bus.jpg").string());
    }

    std::unique_ptr<YoloV8SegPostProcessRknn> processor;
    cnpy::npz_t model_outputs_npz;
    fs::path path_seg_result;
    cv::Mat img;
};

TEST_F(YoloV8SegPostProcessRknnTest, GoldenRun)
{
    const std::vector<Detection> expected_detections = {
        MakeDetectResult(5, 0.9113f, 87, 137, 553, 439),
        MakeDetectResult(0, 0.8998f, 108, 236, 227, 537),
        MakeDetectResult(0, 0.8693f, 211, 241, 283, 508),
        MakeDetectResult(0, 0.8655f, 477, 232, 559, 519),
        MakeDetectResult(0, 0.5403f, 79, 327, 125, 514),
        MakeDetectResult(27, 0.2741f, 248, 284, 259, 310),
    };

    const std::vector<uint8_t> expected_seg_mask = loadUint8VectorFromBinFile(path_seg_result);

    std::vector<std::vector<int>> output_dims = {
        {1, 64, 80, 80},
        {1, 80, 80, 80},
        {1, 1, 80, 80},
        {1, 32, 80, 80},
        {1, 64, 40, 40},
        {1, 80, 40, 40},
        {1, 1, 40, 40},
        {1, 32, 40, 40},
        {1, 64, 20, 20},
        {1, 80, 20, 20},
        {1, 1, 20, 20},
        {1, 32, 20, 20},
        {1, 32, 160, 160},
    };

    std::vector<float> output_scales = {
        0.115401f, 0.003514f, 0.003540f, 0.019863f,
        0.099555f, 0.003555f, 0.003680f, 0.022367f,
        0.074253f, 0.003813f, 0.003922f, 0.019919f,
        0.032336f,
    };

    std::vector<int32_t> output_zps = {
        -61, -128, -128, 27,
        -15, -128, -128, 30,
        -55, -128, -128, 43,
        -119,
    };

    auto model_outputs = deploy_percept::utils::convertNpzToInt8VectorsByPrefix(
        model_outputs_npz, "output_", 13);
    const std::vector<int8_t *> inputs = makeInt8PtrInputs(model_outputs);

    ASSERT_TRUE(processor->run(inputs, 640, 640, output_dims, output_scales, output_zps));

    const auto &result = processor->getResult();
    expectDetectionsMatchGolden(expected_detections, result.detections);
    EXPECT_TRUE(isUint8VectorEqualWithTolerance(expected_seg_mask, result.label_mask, 0.03));

    deploy_percept::utils::drawSegmentationResults(img, result);
    cv::imwrite((output_dir() / "yolov8_seg_rknn_result.jpg").string(), img);
}
