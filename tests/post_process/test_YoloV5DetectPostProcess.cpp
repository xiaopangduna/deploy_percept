#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <fstream>
#include <streambuf>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <filesystem>

#include <yaml-cpp/yaml.h>

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

// 从NPZ文件读取数据的辅助函数
bool readNpzFile(const std::string &filepath,
                 std::vector<int8_t> &output0, std::vector<int> &shape0,
                 std::vector<int8_t> &output1, std::vector<int> &shape1,
                 std::vector<int8_t> &output2, std::vector<int> &shape2)
{
    try
    {
        // 加载NPZ文件
        cnpy::npz_t npzFile = cnpy::npz_load(filepath);

        // 获取output0
        cnpy::NpyArray arr0 = npzFile["output0"];
        if (arr0.word_size != sizeof(int8_t))
        {
            std::cerr << "output0 data type mismatch" << std::endl;
            return false;
        }
        output0.resize(arr0.num_vals);
        std::memcpy(output0.data(), arr0.data<int8_t>(), arr0.num_vals * sizeof(int8_t));

        shape0.assign(arr0.shape.begin(), arr0.shape.end());

        // 获取output1
        cnpy::NpyArray arr1 = npzFile["output1"];
        if (arr1.word_size != sizeof(int8_t))
        {
            std::cerr << "output1 data type mismatch" << std::endl;
            return false;
        }
        output1.resize(arr1.num_vals);
        std::memcpy(output1.data(), arr1.data<int8_t>(), arr1.num_vals * sizeof(int8_t));

        shape1.assign(arr1.shape.begin(), arr1.shape.end());

        // 获取output2
        cnpy::NpyArray arr2 = npzFile["output2"];
        if (arr2.word_size != sizeof(int8_t))
        {
            std::cerr << "output2 data type mismatch" << std::endl;
            return false;
        }
        output2.resize(arr2.num_vals);
        std::memcpy(output2.data(), arr2.data<int8_t>(), arr2.num_vals * sizeof(int8_t));

        shape2.assign(arr2.shape.begin(), arr2.shape.end());

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading NPZ file: " << e.what() << std::endl;
        return false;
    }
}

// 从YAML文件读取参数的辅助函数
bool readParamsYaml(const std::string &filepath,
                    int &model_in_h, int &model_in_w,
                    float &box_conf_threshold, float &nms_threshold,
                    std::vector<int32_t> &qnt_zps,
                    std::vector<float> &qnt_scales)
{
    try
    {
        YAML::Node config = YAML::LoadFile(filepath);

        // 读取基本参数
        model_in_h = config["model_h"].as<int>();
        model_in_w = config["model_w"].as<int>();
        box_conf_threshold = config["box_conf_threshold"].as<float>();
        nms_threshold = config["nms_threshold"].as<float>();

        // 读取量化参数
        if (config["qnt_zps"])
        {
            qnt_zps.clear();
            for (const auto &val : config["qnt_zps"])
            {
                qnt_zps.push_back(val.as<int32_t>());
            }
        }

        if (config["qnt_scales"])
        {
            qnt_scales.clear();
            for (const auto &val : config["qnt_scales"])
            {
                qnt_scales.push_back(val.as<float>());
            }
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading YAML file: " << e.what() << std::endl;
        return false;
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
    std::vector<int8_t> input0, input1, input2;
    std::vector<int> shape0, shape1, shape2;

    // 使用项目根目录作为工作路径
    std::filesystem::path npz_path = "examples/data/yolov5_detect/yolov5_outputs.npz";

    bool success = false;

    if (!npz_path.empty())
    {
        success = readNpzFile(npz_path, input0, shape0, input1, shape1, input2, shape2);
    }

    // 使用filesystem构建参数文件路径
    std::filesystem::path params_path = "examples/data/yolov5_detect/yolov5_params.yaml";

    // 读取参数YAML文件
    int model_in_h, model_in_w;
    float box_conf_threshold = 0.5f, nms_threshold = 0.5f;
    std::vector<int32_t> qnt_zps;
    std::vector<float> qnt_scales;

    bool params_loaded = readParamsYaml(params_path,
                                        model_in_h, model_in_w,
                                        box_conf_threshold, nms_threshold,
                                        qnt_zps, qnt_scales);

    // 准备输入向量
    std::vector<int8_t *> inputs = {
        input0.data(),
        input1.data(),
        input2.data()};

    // 执行处理
    bool result = processor->run(
        inputs,
        model_in_h, model_in_w,
        qnt_zps, qnt_scales);

    // 验证结果
    EXPECT_TRUE(result); // 确保处理成功

    // 获取检测结果
    const auto &result_wrapper = processor->getResult();
    const auto &group = result_wrapper.group; // 从Result结构体中获取ResultGroup

    // 验证group和预期结果是否一致

    EXPECT_TRUE(CompareDetectResultVectors(expected_results, group.results));

    std::filesystem::path results_path = "examples/data/yolov5_detect/yolov5_detect_results.yaml";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // 注册全局环境
    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);

    return RUN_ALL_TESTS();
}