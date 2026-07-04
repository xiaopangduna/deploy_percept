#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "cnpy.h"

#include "deploy_percept/core/post_process/YoloV5SegPostProcess.hpp"
#include "deploy_percept/utils/io.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "deploy_percept/utils/vis_draw.hpp"
#include "tests/test_common/fixtures.hpp"
#include "tests/test_common/paths.hpp"
#include "tests/test_common/utils.hpp"

namespace fs = std::filesystem;

using deploy_percept::Detection;
using deploy_percept::post_process::YoloV5SegPostProcess;
using deploy_percept::utils::loadUint8VectorFromBinFile;
using percept::test::fixture_data;
using percept::test::makeInt8PtrInputs;
using percept::test::output_dir;

class YoloV5SegPostProcessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        fs::create_directories(output_dir());

        processor = std::make_unique<YoloV5SegPostProcess>(YoloV5SegPostProcess::Params{});

        model_outputs_npz = cnpy::npz_load(
            fixture_data("unit/post_process/yolov5_seg/model_outputs.npz").string());

        path_seg_result = fixture_data("unit/post_process/yolov5_seg/mask.bin");
        img = cv::imread(fixture_data("unit/post_process/yolov5_seg/bus.jpg").string());
    }

    std::unique_ptr<YoloV5SegPostProcess> processor;
    cnpy::npz_t model_outputs_npz;
    fs::path path_seg_result;
    cv::Mat img;
};

TEST_F(YoloV5SegPostProcessTest, GoldenRun)
{
    const std::vector<Detection> expected_detections = {
        MakeDetectResult(0, 0.8823f, 213, 239, 284, 516),
        MakeDetectResult(0, 0.8651f, 109, 239, 224, 536),
        MakeDetectResult(0, 0.8416f, 473, 230, 559, 522),
        MakeDetectResult(5, 0.8239f, 97, 134, 548, 461),
        MakeDetectResult(0, 0.5007f, 79, 325, 124, 520),
    };

    const std::vector<uint8_t> expected_seg_mask = loadUint8VectorFromBinFile(path_seg_result);

    const std::vector<std::vector<int>> output_dims = {
        {1, 255, 80, 80},
        {1, 96, 80, 80},
        {1, 255, 40, 40},
        {1, 96, 40, 40},
        {1, 255, 20, 20},
        {1, 96, 20, 20},
        {1, 32, 160, 160},
    };

    const std::vector<float> output_scales = {
        0.003922f, 0.022222f, 0.003922f, 0.023239f, 0.003918f, 0.024074f, 0.022475f,
    };

    const std::vector<int32_t> output_zps = {-128, 20, -128, 29, -128, 32, -116};

    auto model_outputs = deploy_percept::utils::convertNpzToInt8VectorsByPrefix(
        model_outputs_npz, "output_", 7);
    const std::vector<int8_t *> inputs = makeInt8PtrInputs(model_outputs);

    ASSERT_TRUE(processor->run(inputs, 640, 640, output_dims, output_scales, output_zps));

    const auto &result = processor->getResult();
    expectDetectionsMatchGolden(expected_detections, result.detections);
    EXPECT_TRUE(isUint8VectorEqualWithTolerance(expected_seg_mask, result.label_mask, 0.03));

    deploy_percept::utils::drawDetectionResults(img, result);
    cv::imwrite((output_dir() / "yolov5_seg_result.jpg").string(), img);
}
