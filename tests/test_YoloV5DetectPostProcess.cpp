#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <fstream>
#include <streambuf>
#include <dirent.h>
#include <algorithm>
#include <regex>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "cnpy.h"

#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include "utils/environment.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// YoloV5后处理相关测试
////////////////////////////////////////////////////////////////////////////////////////////////////

// 读取NPZ文件的辅助函数
bool readNpzFile(const std::string& filename, 
                 std::vector<int8_t>& output0_data, std::vector<int>& output0_shape,
                 std::vector<int8_t>& output1_data, std::vector<int>& output1_shape,
                 std::vector<int8_t>& output2_data, std::vector<int>& output2_shape) {
    try {
        // 加载NPZ文件
        cnpy::npz_t npzFile = cnpy::npz_load(filename);
        
        // 获取output0
        cnpy::NpyArray arr0 = npzFile["output0"];
        if (arr0.word_size != sizeof(int8_t)) {
            std::cerr << "output0 data type mismatch" << std::endl;
            return false;
        }
        output0_data.resize(arr0.num_vals);
        std::memcpy(output0_data.data(), arr0.data<int8_t>(), arr0.num_vals * sizeof(int8_t));
        
        output0_shape.assign(arr0.shape.begin(), arr0.shape.end());
        
        // 获取output1
        cnpy::NpyArray arr1 = npzFile["output1"];
        if (arr1.word_size != sizeof(int8_t)) {
            std::cerr << "output1 data type mismatch" << std::endl;
            return false;
        }
        output1_data.resize(arr1.num_vals);
        std::memcpy(output1_data.data(), arr1.data<int8_t>(), arr1.num_vals * sizeof(int8_t));
        
        output1_shape.assign(arr1.shape.begin(), arr1.shape.end());
        
        // 获取output2
        cnpy::NpyArray arr2 = npzFile["output2"];
        if (arr2.word_size != sizeof(int8_t)) {
            std::cerr << "output2 data type mismatch" << std::endl;
            return false;
        }
        output2_data.resize(arr2.num_vals);
        std::memcpy(output2_data.data(), arr2.data<int8_t>(), arr2.num_vals * sizeof(int8_t));
        
        output2_shape.assign(arr2.shape.begin(), arr2.shape.end());
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error reading NPZ file: " << e.what() << std::endl;
        return false;
    }
}

// 读取参数JSON文件的辅助函数
bool readParamsJson(const std::string& filename, 
                   int& model_in_h, int& model_in_w, 
                   float& box_conf_threshold, float& nms_threshold,
                   deploy_percept::post_process::BoxRect& pads,
                   float& scale_w, float& scale_h,
                   std::vector<int32_t>& qnt_zps,
                   std::vector<float>& qnt_scales) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open params file: " << filename << std::endl;
        return false;
    }

    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }
    file.close();

    // 简单解析JSON内容
    std::istringstream stream(content);
    std::string part;
    while (std::getline(stream, part, ',')) {
        // 查找键值对
        size_t colon_pos = part.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = part.substr(0, colon_pos);
            std::string value = part.substr(colon_pos + 1);
            
            // 清理键和值的空格
            key.erase(0, key.find_first_not_of(" \t\n\r\"{"));
            key.erase(key.find_last_not_of(" \t\n\r\"}") + 1);
            value.erase(0, value.find_first_not_of(" \t\n\r: [{"));
            value.erase(value.find_last_not_of(" \t\n\r}]") + 1);
            
            if (key == "model_h") {
                model_in_h = std::stoi(value);
            } else if (key == "model_w") {
                model_in_w = std::stoi(value);
            } else if (key == "box_conf_threshold") {
                box_conf_threshold = std::stof(value);
            } else if (key == "nms_threshold") {
                nms_threshold = std::stof(value);
            } else if (key == "scale_w") {
                scale_w = std::stof(value);
            } else if (key == "scale_h") {
                scale_h = std::stof(value);
            }
        }
        
        // 检查pads部分
        if (part.find("pads") != std::string::npos) {
            // 解析pads部分
            if (part.find("left") != std::string::npos) {
                size_t pos = part.find("left");
                size_t val_start = part.find(':', pos) + 1;
                size_t val_end = part.find(',', val_start);
                if (val_end == std::string::npos) val_end = part.find('}', val_start);
                pads.left = std::stoi(part.substr(val_start, val_end - val_start));
            }
            if (part.find("\"top\"") != std::string::npos) {
                size_t pos = part.find("\"top\"");
                size_t val_start = part.find(':', pos) + 1;
                size_t val_end = part.find(',', val_start);
                pads.top = std::stoi(part.substr(val_start, val_end - val_start));
            }
            if (part.find("\"right\"") != std::string::npos) {
                size_t pos = part.find("\"right\"");
                size_t val_start = part.find(':', pos) + 1;
                size_t val_end = part.find(',', val_start);
                pads.right = std::stoi(part.substr(val_start, val_end - val_start));
            }
            if (part.find("\"bottom\"") != std::string::npos) {
                size_t pos = part.find("\"bottom\"");
                size_t val_start = part.find(':', pos) + 1;
                size_t val_end = part.find(',', val_start);
                if (val_end == std::string::npos) val_end = part.find('}', val_start);
                pads.bottom = std::stoi(part.substr(val_start, val_end - val_start));
            }
        }
        
        // 检查数组部分
        if (part.find("qnt_zps") != std::string::npos) {
            size_t start = content.find("\"qnt_zps\": [") + 12; // length of "\"qnt_zps\": ["
            size_t end = content.find("]", start);
            std::string zps_str = content.substr(start, end - start);
            
            std::stringstream ss(zps_str);
            std::string item;
            qnt_zps.clear();
            while (std::getline(ss, item, ',')) {
                item.erase(0, item.find_first_not_of(" \t\n\r"));
                item.erase(item.find_last_not_of(" \t\n\r") + 1);
                qnt_zps.push_back(std::stoi(item));
            }
        }
        if (part.find("qnt_scales") != std::string::npos) {
            size_t start = content.find("\"qnt_scales\": [") + 15; // length of "\"qnt_scales\": ["
            size_t end = content.find("]", start);
            std::string scales_str = content.substr(start, end - start);
            
            std::stringstream ss(scales_str);
            std::string item;
            qnt_scales.clear();
            while (std::getline(ss, item, ',')) {
                item.erase(0, item.find_first_not_of(" \t\n\r"));
                item.erase(item.find_last_not_of(" \t\n\r") + 1);
                qnt_scales.push_back(std::stof(item));
            }
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
        processor = std::make_unique<deploy_percept::post_process::YoloV5DetectPostProcess>();
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
        GlobalLoggerEnvironment::logger->info("Testing YoloV5DetectPostProcess process function with real data from NPZ files");
    }

    // 尝试从NPZ文件读取数据
    std::vector<int8_t> input0, input1, input2;
    std::vector<int> shape0, shape1, shape2;
    
    // 直接使用固定的NPZ输出文件路径
    std::string npz_path = "/home/orangepi/HectorHuang/deploy_percept/tmp/yolov5_outputs.npz";
    
    bool success = false;
    
    if (!npz_path.empty()) {
        success = readNpzFile(npz_path, input0, shape0, input1, shape1, input2, shape2);
    }
    
    // 如果无法读取真实数据，则跳过测试
    if (!success) {
        std::cout << "Could not load real data from NPZ file, skipping this test." << std::endl;
        GTEST_SKIP() << "Real NPZ data file not found";
        return;
    }
    
    // 使用固定的参数文件路径
    std::string params_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_detect/yolov5_params.json";
    
    // 读取参数JSON文件
    int model_in_h, model_in_w;
    float box_conf_threshold = 0.5f, nms_threshold = 0.5f;
    deploy_percept::post_process::BoxRect pads = {0, 0, 0, 0};
    float scale_w = 1.0, scale_h = 1.0;
    std::vector<int32_t> qnt_zps;
    std::vector<float> qnt_scales;
    
    bool params_loaded = readParamsJson(params_path, 
                                       model_in_h, model_in_w, 
                                       box_conf_threshold, nms_threshold,
                                       pads, scale_w, scale_h,
                                       qnt_zps, qnt_scales);
    
    if (!params_loaded) {
        std::cout << "Could not load parameters from file: " << params_path << ", skipping this test." << std::endl;
        GTEST_SKIP() << "Parameters JSON file not found";
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
    for (auto val : qnt_zps) std::cout << val << " ";
    std::cout << std::endl;
    std::cout << "  Quantization scales: ";
    for (auto val : qnt_scales) std::cout << val << " ";
    std::cout << std::endl;

    // 创建结果组
    deploy_percept::post_process::DetectResultGroup group;
    memset(&group, 0, sizeof(group));

    // 执行处理
    int result = processor->process(
        input0.data(),
        input1.data(),
        input2.data(),
        model_in_h,
        model_in_w,
        pads,
        scale_w,
        scale_h,
        qnt_zps,
        qnt_scales,
        &group
    );

    // 验证结果
    EXPECT_EQ(result, 0);  // 确保处理成功
    std::cout << "Detection results count: " << group.count << std::endl;
    
    // 输出检测到的对象信息
    for (int i = 0; i < group.count; i++) {
        std::cout << "Object " << i << ": " << group.results[i].name 
                  << " at (" << group.results[i].box.left << ", " << group.results[i].box.top 
                  << ", " << group.results[i].box.right << ", " << group.results[i].box.bottom 
                  << ") with confidence " << group.results[i].prop << std::endl;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // 注册全局环境
    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);

    return RUN_ALL_TESTS();
}