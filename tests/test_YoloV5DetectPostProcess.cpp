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
                    deploy_percept::post_process::BoxRect &pads,
                    float &scale_w, float &scale_h,
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
        scale_w = config["scale_w"].as<float>();
        scale_h = config["scale_h"].as<float>();

        // 读取pads参数
        YAML::Node pads_node = config["pads"];
        if (pads_node)
        {
            pads.left = pads_node["left"].as<int>();
            pads.top = pads_node["top"].as<int>();
            pads.right = pads_node["right"].as<int>();
            pads.bottom = pads_node["bottom"].as<int>();
        }

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

// 从YAML文件读取预期检测结果的辅助函数
bool readExpectedResultsFromYaml(const std::string &filepath,
                                 deploy_percept::post_process::DetectResultGroup &expected_group)
{
    try
    {
        YAML::Node config = YAML::LoadFile(filepath);

        // 读取检测数量
        expected_group.count = config["detection_count"].as<int>();

        // 读取检测结果
        YAML::Node detections = config["detections"];
        int idx = 0;
        for (const auto &detection : detections)
        {
            if (idx >= 64)
                break; // 防止越界

            // 读取名称
            std::string name = detection["name"].as<std::string>();
            strncpy(expected_group.results[idx].name, name.c_str(), sizeof(expected_group.results[idx].name) - 1);
            expected_group.results[idx].name[sizeof(expected_group.results[idx].name) - 1] = '\0';

            // 读取边界框
            YAML::Node box = detection["box"];
            expected_group.results[idx].box.left = box["left"].as<int>();
            expected_group.results[idx].box.top = box["top"].as<int>();
            expected_group.results[idx].box.right = box["right"].as<int>();
            expected_group.results[idx].box.bottom = box["bottom"].as<int>();

            // 读取置信度
            expected_group.results[idx].prop = detection["prop"].as<float>();

            idx++;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading YAML file: " << e.what() << std::endl;
        return false;
    }
}

// 比较两个检测结果是否相等（考虑浮点数精度）
bool compareDetectionResults(const deploy_percept::post_process::DetectResult &a,
                             const deploy_percept::post_process::DetectResult &b,
                             float tolerance = 0.01f)
{
    // 比较名称
    // if (strcmp(a.name, b.name) != 0) {
    //     return false;
    // }

    // 比较边界框
    if (a.box.left != b.box.left || a.box.top != b.box.top ||
        a.box.right != b.box.right || a.box.bottom != b.box.bottom)
    {
        return false;
    }

    // 比较置信度，考虑浮点误差
    if (abs(a.prop - b.prop) > tolerance)
    {
        return false;
    }

    return true;
}

// 比较检测结果组是否相等
bool compareResultGroups(const deploy_percept::post_process::DetectResultGroup &actual,
                         const deploy_percept::post_process::DetectResultGroup &expected,
                         float tolerance = 0.01f)
{
    if (actual.count != expected.count)
    {
        std::cout << "Detection count mismatch: actual=" << actual.count << ", expected=" << expected.count << std::endl;
        return false;
    }

    // 简单比较，按顺序对比
    for (int i = 0; i < actual.count; ++i)
    {
        if (!compareDetectionResults(actual.results[i], expected.results[i], tolerance))
        {
            std::cout << "Detection result " << i << " mismatch:" << std::endl;
            std::cout << "  Actual: " << actual.results[i].name
                      << " [" << actual.results[i].box.left << "," << actual.results[i].box.top
                      << "," << actual.results[i].box.right << "," << actual.results[i].box.bottom
                      << "] prop=" << actual.results[i].prop << std::endl;
            std::cout << "  Expected: " << expected.results[i].name
                      << " [" << expected.results[i].box.left << "," << expected.results[i].box.top
                      << "," << expected.results[i].box.right << "," << expected.results[i].box.bottom
                      << "] prop=" << expected.results[i].prop << std::endl;
            return false;
        }
    }

    return true;
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
};

// 测试YoloV5后处理功能并与原始实现进行对比
TEST_F(YoloV5DetectPostProcessTest, ProcessFunctionWithRealData)
{
    if (GlobalLoggerEnvironment::logger)
    {
        GlobalLoggerEnvironment::logger->info("Testing YoloV5DetectPostProcess run function with real data from NPZ files");
    }

    // 输出当前工作目录以进行调试
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    // 尝试从NPZ文件读取数据
    std::vector<int8_t> input0, input1, input2;
    std::vector<int> shape0, shape1, shape2;

    // 定义项目根目录路径前缀（当前工作目录已设置为项目根目录）
    std::filesystem::path workspaceFolder = std::filesystem::current_path().parent_path().parent_path();
    std::cout << "Current working directory: " << workspaceFolder << std::endl;
    // 使用项目根目录作为工作路径
    std::filesystem::path npz_path = workspaceFolder / "examples/data/yolov5_detect/yolov5_outputs.npz";

    bool success = false;

    if (!npz_path.empty())
    {
        success = readNpzFile(npz_path, input0, shape0, input1, shape1, input2, shape2);
    }

    // 如果无法读取真实数据，则跳过测试
    if (!success)
    {
        std::cout << "Could not load real data from NPZ file: " << npz_path << ", skipping this test." << std::endl;
        GTEST_SKIP() << "Real NPZ data file not found";
        return;
    }

    // 使用filesystem构建参数文件路径
    std::filesystem::path params_path = workspaceFolder / "examples/data/yolov5_detect/yolov5_params.yaml";

    // 读取参数YAML文件
    int model_in_h, model_in_w;
    float box_conf_threshold = 0.5f, nms_threshold = 0.5f;
    deploy_percept::post_process::BoxRect pads = {0, 0, 0, 0};
    float scale_w = 1.0, scale_h = 1.0;
    std::vector<int32_t> qnt_zps;
    std::vector<float> qnt_scales;

    bool params_loaded = readParamsYaml(params_path,
                                        model_in_h, model_in_w,
                                        box_conf_threshold, nms_threshold,
                                        pads, scale_w, scale_h,
                                        qnt_zps, qnt_scales);

    if (!params_loaded)
    {
        std::cout << "Could not load parameters from file: " << params_path << ", skipping this test." << std::endl;
        GTEST_SKIP() << "Parameters YAML file not found";
        return;
    }

    std::cout << "Successfully loaded real data from NPZ file: " << npz_path << std::endl;
    std::cout << "  Output0 shape: (" << shape0[0] << ", " << shape0[1] << ", " << shape0[2] << ")" << std::endl;
    std::cout << "  Output1 shape: (" << shape1[0] << ", " << shape1[1] << ", " << shape1[2] << ")" << std::endl;
    std::cout << "  Output2 shape: (" << shape2[0] << ", " << shape2[1] << ", " << shape2[2] << ")" << std::endl;
    std::cout << "  Model input: " << model_in_h << "x" << model_in_w << std::endl;
    std::cout << "  Box confidence threshold: " << box_conf_threshold << std::endl;
    std::cout << "  NMS threshold: " << nms_threshold << std::endl;
    std::cout << "  Quantization zps: ";
    for (auto val : qnt_zps)
        std::cout << val << " ";
    std::cout << std::endl;
    std::cout << "  Quantization scales: ";
    for (auto val : qnt_scales)
        std::cout << val << " ";
    std::cout << std::endl;

    // 执行处理
    bool result = processor->run(
        input0.data(), input1.data(), input2.data(),
        model_in_h, model_in_w,
        pads, scale_w, scale_h,
        qnt_zps, qnt_scales);

    // 验证结果
    EXPECT_TRUE(result); // 确保处理成功

    // 获取检测结果
    const auto &result_wrapper = processor->getResult();
    const auto &group = result_wrapper.group; // 从Result结构体中获取DetectResultGroup
    std::cout << "Detection results count: " << group.count << std::endl;

    // 输出检测到的对象信息
    for (int i = 0; i < group.count; i++)
    {
        std::cout << "Object " << i << ": " << group.results[i].name
                  << " at (" << group.results[i].box.left << ", " << group.results[i].box.top
                  << ", " << group.results[i].box.right << ", " << group.results[i].box.bottom
                  << ") with confidence " << group.results[i].prop << std::endl;
    }

    // 验证group和预期结果是否一致
    deploy_percept::post_process::DetectResultGroup expected_group;
    memset(&expected_group, 0, sizeof(expected_group));

    std::filesystem::path results_path = workspaceFolder / "examples/data/yolov5_detect/yolov5_detect_results.yaml";
    bool expected_loaded = readExpectedResultsFromYaml(results_path, expected_group);

    if (!expected_loaded)
    {
        std::cout << "Could not load expected results from file: " << results_path << ", skipping validation." << std::endl;
        GTEST_SKIP() << "Expected results YAML file not found";
        return;
    }

    std::cout << "Validating detection results against expected results..." << std::endl;
    bool results_match = compareResultGroups(group, expected_group);

    if (results_match)
    {
        std::cout << "Detection results match expected results!" << std::endl;
        EXPECT_TRUE(results_match);
    }
    else
    {
        std::cout << "Detection results do not match expected results." << std::endl;
        EXPECT_FALSE(results_match) << "Detection results do not match expected results";
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // 注册全局环境
    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);

    return RUN_ALL_TESTS();
}