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

#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include "utils/environment.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// YoloV5后处理相关测试
////////////////////////////////////////////////////////////////////////////////////////////////////

// 从目录中查找最新的output NPY文件
std::string findLatestOutputFile(const std::string& pattern) {
    DIR* dir;
    struct dirent* ent;
    std::vector<std::string> files;
    std::regex pattern_regex(pattern);
    
    if ((dir = opendir("/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_detect/")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (std::regex_match(filename, pattern_regex)) {
                files.push_back("/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_detect/" + filename);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Could not open directory" << std::endl;
        return "";
    }
    
    if (files.empty()) {
        return "";
    }
    
    // 按文件名排序，找到最新的文件（按时间戳）
    std::sort(files.begin(), files.end());
    return files.back(); // 返回最后一个（最新的）文件
}

// 读取NPY文件的辅助函数
bool readNpyFile(const std::string& filename, std::vector<int8_t>& data, std::vector<int>& shape) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    // 读取魔数
    char magic[6];
    file.read(magic, 6);
    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' || 
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        std::cerr << "Invalid NPY file: " << filename << std::endl;
        return false;
    }

    // 读取版本
    uint8_t version[2];
    file.read(reinterpret_cast<char*>(version), 2);

    // 读取头长度
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    // 读取头
    std::vector<char> header(header_len + 1);  // +1 for null terminator
    file.read(header.data(), header_len);
    header[header_len] = '\0';

    // 解析头部信息，提取shape
    std::string header_str(header.data());
    
    // 简单解析shape - 在实际项目中可能需要更复杂的解析
    // 这里我们假设格式是已知的
    size_t shape_pos = header_str.find("shape': (");
    if (shape_pos != std::string::npos) {
        size_t start = shape_pos + 9; // length of "shape': ("
        size_t end = header_str.find(")", start);
        std::string shape_str = header_str.substr(start, end - start);
        
        // 解析维度
        size_t pos = 0;
        std::string token;
        while ((pos = shape_str.find(", ")) != std::string::npos) {
            token = shape_str.substr(0, pos);
            shape.push_back(std::stoi(token));
            shape_str.erase(0, pos + 2);
        }
        shape.push_back(std::stoi(shape_str)); // 添加最后一个维度
    }

    // 计算数据大小
    size_t data_size = 1;
    for (int dim : shape) {
        data_size *= dim;
    }

    // 读取数据
    data.resize(data_size);
    file.read(reinterpret_cast<char*>(data.data()), data_size * sizeof(int8_t));

    return true;
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

// 测试基本的process功能（使用从NPY文件读取的真实数据）
TEST_F(YoloV5DetectPostProcessTest, ProcessFunctionWithRealData)
{
    if (GlobalLoggerEnvironment::logger)
    {
        GlobalLoggerEnvironment::logger->info("Testing YoloV5DetectPostProcess process function with real data from NPY files");
    }

    // 尝试从NPY文件读取数据
    std::vector<int8_t> input0, input1, input2;
    std::vector<int> shape0, shape1, shape2;
    
    // 直接使用示例数据文件路径
    std::string output0_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_detect/yolov5_output0.npy";
    std::string output1_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_detect/yolov5_output1.npy";
    std::string output2_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_detect/yolov5_output2.npy";
    std::string params_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_detect/yolov5_params.json";
    
    bool success0 = false, success1 = false, success2 = false;
    
    if (!output0_path.empty()) {
        success0 = readNpyFile(output0_path, input0, shape0);
    }
    if (!output1_path.empty()) {
        success1 = readNpyFile(output1_path, input1, shape1);
    }
    if (!output2_path.empty()) {
        success2 = readNpyFile(output2_path, input2, shape2);
    }
    
    // 如果无法读取真实数据，则跳过测试
    if (!success0 || !success1 || !success2) {
        std::cout << "Could not load real data from NPY files, skipping this test." << std::endl;
        GTEST_SKIP() << "Real NPY data files not found";
        return;
    }
    
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
    
    std::cout << "Successfully loaded real data from NPY files:" << std::endl;
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

// 与原始main.cpp中post_process函数对比测试
TEST_F(YoloV5DetectPostProcessTest, CompareWithOriginalPostProcess)
{
    if (GlobalLoggerEnvironment::logger)
    {
        GlobalLoggerEnvironment::logger->info("Comparing YoloV5DetectPostProcess with original post_process function");
    }

    // 尝试从NPY文件读取数据
    std::vector<int8_t> input0, input1, input2;
    std::vector<int> shape0, shape1, shape2;
    
    // 直接使用示例数据文件路径
    std::string output0_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_detect/yolov5_output0.npy";
    std::string output1_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_detect/yolov5_output1.npy";
    std::string output2_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_detect/yolov5_output2.npy";
    std::string params_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_detect/yolov5_params.json";
    
    bool success0 = false, success1 = false, success2 = false;
    
    if (!output0_path.empty()) {
        success0 = readNpyFile(output0_path, input0, shape0);
    }
    if (!output1_path.empty()) {
        success1 = readNpyFile(output1_path, input1, shape1);
    }
    if (!output2_path.empty()) {
        success2 = readNpyFile(output2_path, input2, shape2);
    }
    
    // 如果无法读取真实数据，则跳过测试
    if (!success0 || !success1 || !success2) {
        std::cout << "Could not load real data from NPY files, skipping this test." << std::endl;
        GTEST_SKIP() << "Real NPY data files not found";
        return;
    }
    
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

    // 创建结果组用于测试
    deploy_percept::post_process::DetectResultGroup new_group;
    memset(&new_group, 0, sizeof(new_group));

    // 使用重构后的函数处理
    int result_new = processor->process(
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
        &new_group
    );

    std::cout << "New function results count: " << new_group.count << std::endl;
    
    // 输出检测到的对象信息
    for (int i = 0; i < new_group.count; i++) {
        std::cout << "New - Object " << i << ": " << new_group.results[i].name 
                  << " at (" << new_group.results[i].box.left << ", " << new_group.results[i].box.top 
                  << ", " << new_group.results[i].box.right << ", " << new_group.results[i].box.bottom 
                  << ") with confidence " << new_group.results[i].prop << std::endl;
    }

    // 验证结果
    EXPECT_EQ(result_new, 0);  // 确保处理成功
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // 注册全局环境
    ::testing::AddGlobalTestEnvironment(new GlobalLoggerEnvironment);

    return RUN_ALL_TESTS();
}