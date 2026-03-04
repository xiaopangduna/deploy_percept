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

        fs::path path_model_outputs_npz = "examples/data/yolov5_seg/yolov5_seg_output.npz";
        model_outputs_npz = cnpy::npz_load(path_model_outputs_npz);

        path_seg_result = "examples/data/yolov5_seg/seg_mask_0.bin";
    }

    void TearDown() override
    {
    }

    std::unique_ptr<deploy_percept::post_process::YoloV5SegPostProcess> processor;
    cnpy::npz_t model_outputs_npz;
    fs::path path_seg_result;
    std::vector<uint8_t> expected_seg_mask;
};

TEST_F(YoloV5SegPostProcessTest, run)
{
    std::vector<DetectionObject> expected_results = {
        MakeDetectResult(5, "class_5", 0.9113f, 87, 137, 553, 439),
        MakeDetectResult(0, "class_0", 0.8998f, 108, 236, 227, 537),
        MakeDetectResult(0, "class_0", 0.8693f, 211, 241, 283, 508),
        MakeDetectResult(0, "class_0", 0.8655f, 477, 232, 559, 519),
        MakeDetectResult(0, "class_0", 0.5403f, 79, 327, 125, 514),
        MakeDetectResult(27, "class_27", 0.2741f, 248, 284, 259, 310)};

    std::vector<uint8_t> expected_seg_mask;
    expected_seg_mask = LoadUint8VectorFromBinFile(path_seg_result);

    std::vector<std::vector<int>> output_dims;
    std::vector<float> output_scales;
    std::vector<int32_t> output_zps;

    output_dims = {
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
        {1, 32, 160, 160}};

    output_scales = {
        0.115401f, 0.003514f, 0.003540f, 0.019863f,
        0.099555f, 0.003555f, 0.003680f, 0.022367f,
        0.074253f, 0.003813f, 0.003922f, 0.019919f,
        0.032336f};

    output_zps = {
        -61, -128, -128, 27,
        -15, -128, -128, 30,
        -55, -128, -128, 43,
        -119};

    auto model_outputs = deploy_percept::utils::convertNpzToInt8VectorsByPrefix(model_outputs_npz, "output_", 13); // 使用utils命名空间
    std::vector<int8_t *> inputs = {
        model_outputs[0].data(),
        model_outputs[1].data(),
        model_outputs[2].data(),
        model_outputs[3].data(),
        model_outputs[4].data(),
        model_outputs[5].data(),
        model_outputs[6].data(),
        model_outputs[7].data(),
        model_outputs[8].data(),
        model_outputs[9].data(),
        model_outputs[10].data(),
        model_outputs[11].data(),
        model_outputs[12].data()};

    bool success = processor->run(inputs, 640, 640,
                                  output_dims, output_scales, output_zps);

    const auto &result = processor->getResult();

    const auto &result_group = result.group;

    EXPECT_TRUE(isDetectionObjectVectorEqualWithinTolerance(expected_results, result_group.detection_objects));

    const auto &actual_results = processor->getResult().group;

    // 使用修改后的函数比较一维分割掩码
    // EXPECT_TRUE(CompareSegmentationMaskVectors(expected_seg_mask, actual_results.segmentation_masks));

    std::string input_path = "apps/yolov8_seg_rknn/bus.jpg";
    cv::Mat orig_img = cv::imread(input_path, 1);
    cv::Mat result_img = orig_img.clone();
    processor->drawDetectionResults(result_img, result_group);
    std::string computed_out_path = "apps/yolov8_seg_rknn/yolov8_seg_result.jpg";
    cv::imwrite(computed_out_path, result_img);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}