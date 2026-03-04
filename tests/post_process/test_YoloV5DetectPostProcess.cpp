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
#include "utils/environment.hpp"
#include "tests/test_common/compare.hpp"
using namespace deploy_percept::post_process;
namespace fs = std::filesystem;
////////////////////////////////////////////////////////////////////////////////////////////////////
// YoloV5后处理相关测试
////////////////////////////////////////////////////////////////////////////////////////////////////

// 从NPZ文件读取数据的辅助函数，返回std::vector<int8_t*>以便直接输入processor->run
std::vector<int8_t*> readNpzFile(const std::string &filepath, bool &success)
{
    std::vector<int8_t*> result(3, nullptr);  // 初始化3个nullptr
    success = false;
    
    try
    {
        // 加载NPZ文件
        cnpy::npz_t npzFile = cnpy::npz_load(filepath);

        // 获取output0
        cnpy::NpyArray arr0 = npzFile["output0"];
        if (arr0.word_size != sizeof(int8_t))
        {
            std::cerr << "output0 data type mismatch" << std::endl;
            return result;
        }
        result[0] = new int8_t[arr0.num_vals];
        std::memcpy(result[0], arr0.data<int8_t>(), arr0.num_vals * sizeof(int8_t));

        // 获取output1
        cnpy::NpyArray arr1 = npzFile["output1"];
        if (arr1.word_size != sizeof(int8_t))
        {
            std::cerr << "output1 data type mismatch" << std::endl;
            // 清理已分配的内存
            delete[] result[0];
            result[0] = nullptr;
            return result;
        }
        result[1] = new int8_t[arr1.num_vals];
        std::memcpy(result[1], arr1.data<int8_t>(), arr1.num_vals * sizeof(int8_t));

        // 获取output2
        cnpy::NpyArray arr2 = npzFile["output2"];
        if (arr2.word_size != sizeof(int8_t))
        {
            std::cerr << "output2 data type mismatch" << std::endl;
            // 清理已分配的内存
            delete[] result[0];
            delete[] result[1];
            result[0] = nullptr;
            result[1] = nullptr;
            return result;
        }
        result[2] = new int8_t[arr2.num_vals];
        std::memcpy(result[2], arr2.data<int8_t>(), arr2.num_vals * sizeof(int8_t));

        success = true;
        return result;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading NPZ file: " << e.what() << std::endl;
        // 清理已分配的内存
        for (int i = 0; i < 3; ++i) {
            if (result[i] != nullptr) {
                delete[] result[i];
                result[i] = nullptr;
            }
        }
        return result;
    }
}

// 辅助函数：释放readNpzFile分配的内存
void freeNpzData(std::vector<int8_t*> &data)
{
    for (int i = 0; i < data.size(); ++i) {
        if (data[i] != nullptr) {
            delete[] data[i];
            data[i] = nullptr;
        }
    }
}

// YoloV5检测后处理测试夹具类
class YoloV5DetectPostProcessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (GlobalLoggerEnvironment::logger)
        {
            GlobalLoggerEnvironment::logger->info("Setting up YoloV5DetectPostProcess test");
        }

        // 初始化一个基本的YoloV5DetectPostProcess实例
        processor = std::make_unique<deploy_percept::post_process::YoloV5DetectPostProcess>(
            deploy_percept::post_process::YoloV5DetectPostProcess::Params{});
    }

    void TearDown() override
    {
        processor.reset();
        if (GlobalLoggerEnvironment::logger)
        {
            GlobalLoggerEnvironment::logger->info("Tearing down YoloV5DetectPostProcess test");
        }
    }

    std::unique_ptr<deploy_percept::post_process::YoloV5DetectPostProcess> processor;
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
    std::filesystem::path npz_path = "examples/data/yolov5_detect/yolov5_outputs.npz";
    
    bool success = false;
    std::vector<int8_t*> inputs = readNpzFile(npz_path.string(), success);

    // 硬编码参数值（替代从YAML文件读取）
    int model_in_h = 640;
    int model_in_w = 640;
    
    std::vector<int32_t> qnt_zps = {-128, -128, -128};
    std::vector<float> qnt_scales = {0.00392157f, 0.00392157f, 0.00392157f};

    // 执行处理（现在可以直接使用inputs，因为它是std::vector<int8_t*>类型）
    bool result = processor->run(
        inputs,
        model_in_h,
        model_in_w,
        qnt_zps,
        qnt_scales);

    // 清理内存
    freeNpzData(inputs);

    ASSERT_TRUE(result) << "Processing failed: " << processor->getResult().message;

    // 获取结果并验证
    const auto& detection_result = processor->getResult().group;
    
    // 验证检测结果数量
    EXPECT_EQ(detection_result.count, static_cast<int>(expected_results.size())) 
        << "Expected " << expected_results.size() << " detections, but got " << detection_result.count;

    // 验证每个检测结果
    for (int i = 0; i < std::min(detection_result.count, static_cast<int>(expected_results.size())); ++i) {
        const auto& actual = detection_result.results[i];
        const auto& expected = expected_results[i];
        
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