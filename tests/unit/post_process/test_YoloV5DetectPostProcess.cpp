#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <filesystem>

#include "cnpy.h"

#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/types.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "deploy_percept/utils/vis_draw.hpp"
#include "utils/environment.hpp"
#include "tests/test_common/utils.hpp"
#include "tests/test_common/paths.hpp"

namespace fs = std::filesystem;

using namespace deploy_percept::post_process;
using percept::test::app_data;
using percept::test::output_dir;

class YoloV5DetectPostProcessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        fs::create_directories(output_dir());

        YoloV5DetectPostProcess::Params params;
        processor = std::make_unique<YoloV5DetectPostProcess>(params);

        model_outputs_npz = cnpy::npz_load(
            app_data("yolov5_detect_rknn/yolov5_detect_result_model_outputs.npz"));

        img = cv::imread(app_data("yolov5_detect_rknn/bus.jpg"));

        path_save_img_with_detect_result = (output_dir() / "yolov5_detect_result.jpg").string();
    }

    void TearDown() override
    {
    }

    std::unique_ptr<YoloV5DetectPostProcess> processor;
    cnpy::npz_t model_outputs_npz;

    cv::Mat img;
    std::string path_save_img_with_detect_result;
};
TEST_F(YoloV5DetectPostProcessTest, run)
{
    std::vector<DetectionObject> expected_results = {
        MakeDetectResult(0, "class_0", 0.3495f, 209, 243, 286, 510),
        MakeDetectResult(0, "class_0", 0.2487f, 479, 238, 560, 526),
        MakeDetectResult(0, "class_0", 0.3010f, 109, 238, 231, 534),
        MakeDetectResult(5, "class_5", 0.3672f, 91, 129, 555, 464),
        MakeDetectResult(0, "class_0", 0.5035f, 79, 353, 121, 517)};


    auto model_outputs = deploy_percept::utils::convertNpzToInt8VectorsByPrefix(model_outputs_npz, "output_", 3); 

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

    deploy_percept::utils::drawDetectionResults(img, detection_result);

    cv::imwrite(path_save_img_with_detect_result.c_str(), img);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);

    return RUN_ALL_TESTS();
}