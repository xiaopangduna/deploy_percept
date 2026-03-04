#include <gtest/gtest.h>
#include <vector>
#include <filesystem>

#include "cnpy.h"

#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "utils/environment.hpp"
#include "tests/test_common/utils.hpp"

namespace fs = std::filesystem;

using namespace deploy_percept::post_process;

class YoloV5DetectPostProcessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        YoloV5DetectPostProcess::Params params;
        processor = std::make_unique<YoloV5DetectPostProcess>(params);

        fs::path path_model_output_npz = "apps/yolov5_detect_rknn/yolov5_detect_result_model_outputs.npz";
        model_outputs_npz = cnpy::npz_load(path_model_output_npz);

        fs::path path_img = "apps/yolov5_detect_rknn/bus.jpg";
        img = cv::imread(path_img);
        
        path_save_img_with_detect_result = "tmp/yolov5_detect_result.jpg";
    }

    void TearDown() override
    {
    }

    std::unique_ptr<YoloV5DetectPostProcess> processor;
    cnpy::npz_t model_outputs_npz;

    cv::Mat img;
    fs::path path_save_img_with_detect_result;
};
TEST_F(YoloV5DetectPostProcessTest, run)
{
    std::vector<DetectionObject> expected_results = {
        MakeDetectResult(0, "class_0", 0.8797f, 209, 243, 286, 510),
        MakeDetectResult(0, "class_0", 0.8706f, 479, 238, 560, 526),
        MakeDetectResult(0, "class_0", 0.8398f, 109, 238, 231, 534),
        MakeDetectResult(5, "class_5", 0.6920f, 91, 129, 555, 464),
        MakeDetectResult(0, "class_0", 0.3010f, 79, 353, 121, 517)};


    auto model_outputs = deploy_percept::utils::convertNpzToInt8VectorsByPrefix(model_outputs_npz, "output", 3); // 使用utils命名空间

    int model_in_h = 640;
    int model_in_w = 640;

    std::vector<int32_t> qnt_zps = {-128, -128, -128};
    std::vector<float> qnt_scales = {0.00392157f, 0.00392157f, 0.00392157f};

    std::vector<int8_t *> inputs = {
        model_outputs[0].data(),
        model_outputs[1].data(),
        model_outputs[2].data()};

    ASSERT_TRUE(processor->run(inputs, model_in_h, model_in_w, qnt_zps, qnt_scales));

    auto &detection_result = processor->getResult().group;

    EXPECT_TRUE(isDetectionObjectVectorEqualWithinTolerance(expected_results, detection_result.detection_objects));

    processor->drawDetectionResults(img, detection_result);

    cv::imwrite(path_save_img_with_detect_result.c_str(), img);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // 注册全局环境
    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);

    return RUN_ALL_TESTS();
}