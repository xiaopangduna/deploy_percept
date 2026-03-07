#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>

#include "cnpy.h"

#include "deploy_percept/post_process/YoloV5SegPostProcess.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "deploy_percept/utils/io.hpp"
#include "tests/test_common/utils.hpp"

namespace fs = std::filesystem;

using namespace deploy_percept::post_process;
using namespace deploy_percept::utils;

class YoloV5SegPostProcessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        YoloV5SegPostProcess::Params params;
        processor = std::make_unique<YoloV5SegPostProcess>(params);

        fs::path path_model_outputs_npz = "apps/yolov5_seg_rknn/yolov5_seg_result_model_outputs.npz";
        model_outputs_npz = cnpy::npz_load(path_model_outputs_npz);

        path_seg_result = "apps/yolov5_seg_rknn/yolov5_seg_result_mask.bin";

        fs::path path_img = "apps/yolov5_seg_rknn/bus.jpg";
        img = cv::imread(path_img);

        path_save_img_with_detect_result = "tmp/yolov5_seg_result.jpg";
    }

    void TearDown() override
    {
    }

    std::unique_ptr<YoloV5SegPostProcess> processor;
    cnpy::npz_t model_outputs_npz;
    std::string path_seg_result;
    cv::Mat img;
    std::string path_save_img_with_detect_result;
};

TEST_F(YoloV5SegPostProcessTest, run)
{
    std::vector<DetectionObject> expected_results = {
        MakeDetectResult(0, "class_0", 0.8823f, 213, 239, 284, 516),
        MakeDetectResult(0, "class_0", 0.8651f, 109, 239, 224, 536),
        MakeDetectResult(0, "class_0", 0.8416f, 473, 230, 559, 522),
        MakeDetectResult(5, "class_5", 0.8239f, 97, 134, 548, 461),
        MakeDetectResult(0, "class_0", 0.5007f, 79, 325, 124, 520)};

    std::vector<uint8_t> expected_seg_mask = loadUint8VectorFromBinFile(path_seg_result);

    std::vector<std::vector<int>> output_dims;
    std::vector<float> output_scales;
    std::vector<int32_t> output_zps;

    output_dims = {
        {1, 255, 80, 80},
        {1, 96, 80, 80},
        {1, 255, 40, 40},
        {1, 96, 40, 40},
        {1, 255, 20, 20},
        {1, 96, 20, 20},
        {1, 32, 160, 160}};

    output_scales = {0.003922f, 0.022222f, 0.003922f, 0.023239f, 0.003918f, 0.024074f, 0.022475f};

    output_zps = {-128, 20, -128, 29, -128, 32, -116};

    auto model_outputs = deploy_percept::utils::convertNpzToInt8VectorsByPrefix(model_outputs_npz, "output_", 7);
    std::vector<int8_t *> inputs = {
        model_outputs[0].data(),
        model_outputs[1].data(),
        model_outputs[2].data(),
        model_outputs[3].data(),
        model_outputs[4].data(),
        model_outputs[5].data(),
        model_outputs[6].data()};

    bool success = processor->run(inputs, 640, 640,
                                  output_dims, output_scales, output_zps);

    const auto &result = processor->getResult().group;

    EXPECT_TRUE(isDetectionObjectVectorEqualWithinTolerance(expected_results, result.detection_objects));

    const auto &actual_results = processor->getResult().group;

    EXPECT_TRUE(isUint8VectorEqualWithTolerance(expected_seg_mask, actual_results.segmentation_mask, 0.03));

    // std::filesystem::path path_save_mask_bin = "tmp/yolov5_seg_mask_test.bin";
    // saveUint8VectorToBinFile(result.segmentation_mask, path_save_mask_bin);
    // std::filesystem::path path_1 = "/home/orangepi/HectorHuang/deploy_percept/tmp/yolov5_seg_mask.bin";
    // std::vector<uint8_t> expected_seg_mask_1 = loadUint8VectorFromBinFile(path_1);
    // EXPECT_TRUE(isUint8VectorEqual(expected_seg_mask_1, expected_seg_mask));

    EXPECT_TRUE(isUint8VectorEqual(expected_seg_mask, actual_results.segmentation_mask));

    processor->drawDetectionResults(img, result);
    cv::imwrite(path_save_img_with_detect_result, img);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}