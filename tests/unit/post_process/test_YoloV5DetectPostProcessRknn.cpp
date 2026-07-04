#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <filesystem>

#include "cnpy.h"

#include "deploy_percept/core/post_process/YoloV5DetectPostProcessRknn.hpp"
#include "deploy_percept/common/types.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "deploy_percept/utils/vis_draw.hpp"
#include "utils/environment.hpp"
#include "tests/test_common/utils.hpp"
#include "tests/test_common/paths.hpp"

namespace fs = std::filesystem;

using namespace deploy_percept::post_process;
using deploy_percept::Detection;
using percept::test::fixture_data;
using percept::test::output_dir;

class YoloV5DetectPostProcessRknnTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        fs::create_directories(output_dir());

        YoloV5DetectPostProcessRknn::Params params;
        processor = std::make_unique<YoloV5DetectPostProcessRknn>(params);

        model_outputs_npz = cnpy::npz_load(
            fixture_data("unit/post_process/yolov5_detect/model_outputs.npz"));

        img = cv::imread(fixture_data("unit/post_process/yolov5_detect/bus.jpg"));

        path_save_img_with_detect_result = (output_dir() / "yolov5_detect_result.jpg").string();
    }

    void TearDown() override
    {
    }

    std::unique_ptr<YoloV5DetectPostProcessRknn> processor;
    cnpy::npz_t model_outputs_npz;

    cv::Mat img;
    std::string path_save_img_with_detect_result;
};
TEST_F(YoloV5DetectPostProcessRknnTest, run)
{
    std::vector<Detection> expected_results = {
        MakeDetectResult(0, 0.8797f, 209, 243, 286, 510),
        MakeDetectResult(0, 0.8706f, 479, 238, 560, 526),
        MakeDetectResult(0, 0.8398f, 109, 238, 231, 534),
        MakeDetectResult(5, 0.6920f, 91, 129, 555, 464),
        MakeDetectResult(0, 0.3010f, 79, 353, 121, 517)};

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

    EXPECT_TRUE(isDetectionVectorEqualWithinTolerance(expected_results, detection_result.detections));

    // deploy_percept::utils::drawDetectionResults(img, detection_result);

    cv::imwrite(path_save_img_with_detect_result.c_str(), img);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);

    return RUN_ALL_TESTS();
}
