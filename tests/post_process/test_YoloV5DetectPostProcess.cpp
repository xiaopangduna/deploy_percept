#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <fstream>
#include <streambuf>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <filesystem>

#include "cnpy.h"

#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include "deploy_percept/utils/npy.hpp"
#include "utils/environment.hpp"
#include "tests/test_common/compare.hpp"
using namespace deploy_percept::post_process;
namespace fs = std::filesystem;

// YoloV5检测后处理测试夹具类
class YoloV5DetectPostProcessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        YoloV5DetectPostProcess::Params params;
        processor = std::make_unique<YoloV5DetectPostProcess>(params);
    }

    void TearDown() override
    {
    }

    std::unique_ptr<YoloV5DetectPostProcess> processor;
    cnpy::npz_t model_outputs_npz;
    static DetectionObject MakeDetectResult(int cls_id, const char *name_str, float conf,
                                            int x1, int y1, int x2, int y2)
    {
        DetectionObject res;
        res.cls_id = cls_id;
        res.prop = conf;
        res.box = {x1, y1, x2, y2};
        strncpy(res.name, name_str, sizeof(res.name) - 1);
        res.name[sizeof(res.name) - 1] = '\0';
        return res;
    }
};

// 测试YoloV5后处理功能并与原始实现进行对比
TEST_F(YoloV5DetectPostProcessTest, ProcessFunctionWithRealData)
{
    std::vector<DetectionObject> expected_results = {
        MakeDetectResult(0, "class_0", 0.8797f, 209, 243, 286, 510),
        MakeDetectResult(0, "class_0", 0.8706f, 479, 238, 560, 526),
        MakeDetectResult(0, "class_0", 0.8398f, 109, 238, 231, 534),
        MakeDetectResult(5, "class_5", 0.6920f, 91, 129, 555, 464),
        MakeDetectResult(0, "class_0", 0.3010f, 79, 353, 121, 517)};

    // 尝试从NPZ文件读取数据
    std::filesystem::path npz_path = "apps/yolov5_detect_rknn/yolov5_outputs.npz";

    bool success = false;
    auto input_data = deploy_percept::utils::readNpzFile(npz_path.string(), success); // 使用utils命名空间

    // 硬编码参数值（替代从YAML文件读取）
    int model_in_h = 640;
    int model_in_w = 640;
    float box_conf_threshold = 0.5f;
    float nms_threshold = 0.5f;

    std::vector<int32_t> qnt_zps = {-128, -128, -128};
    std::vector<float> qnt_scales = {0.00392157f, 0.00392157f, 0.00392157f};

    // 准备输入向量（从vector<vector>转换为vector<int8_t*>）
    std::vector<int8_t *> inputs = {
        input_data[0].data(),
        input_data[1].data(),
        input_data[2].data()};

    // 执行处理（现在可以直接使用inputs，因为它是std::vector<int8_t*>类型）
    bool result = processor->run(
        inputs,
        model_in_h,
        model_in_w,
        qnt_zps,
        qnt_scales);

    // 获取结果并验证
    const auto &detection_result = processor->getResult().group;

    // 验证每个检测结果
    for (int i = 0; i < std::min(detection_result.count, static_cast<int>(expected_results.size())); ++i)
    {
        const auto &actual = detection_result.results[i];
        const auto &expected = expected_results[i];

        EXPECT_EQ(actual.cls_id, expected.cls_id) << "Detection " << i << " class ID mismatch";
        EXPECT_NEAR(actual.prop, expected.prop, 1e-3) << "Detection " << i << " confidence mismatch";
        EXPECT_EQ(actual.box.left, expected.box.left) << "Detection " << i << " box left mismatch";
        EXPECT_EQ(actual.box.top, expected.box.top) << "Detection " << i << " box top mismatch";
        EXPECT_EQ(actual.box.right, expected.box.right) << "Detection " << i << " box right mismatch";
        EXPECT_EQ(actual.box.bottom, expected.box.bottom) << "Detection " << i << " box bottom mismatch";
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // 注册全局环境
    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);

    return RUN_ALL_TESTS();
}