#include <iostream>
#include <stdio.h>
#include <string>
#include <memory>
#include <vector>
#include <sys/time.h>
#include <fstream>
#include <chrono>

#include "rknn_api.h"

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "cnpy.h"

#include <yaml-cpp/yaml.h>

#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include "deploy_percept/engine/RknnEngine.hpp"

#define LABEL_NALE_TXT_PATH "/home/orangepi/HectorHuang/deploy_percept/apps/yolov5_seg_rknn/coco_80_labels_list.txt"

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

// 比较img.data和examples/data/yolov5_seg/input_image.bin内容是否一致的函数
bool compareImageData(const cv::Mat& img, const std::string& binFilePath) {
    // 读取二进制文件
    std::ifstream file(binFilePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "无法打开二进制文件: " << binFilePath << std::endl;
        return false;
    }
    
    // 获取文件大小
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 验证图像数据大小是否匹配
    size_t imageDataSize = img.total() * img.elemSize();
    if (imageDataSize != static_cast<size_t>(fileSize)) {
        std::cerr << "图像数据大小不匹配!" << std::endl;
        std::cerr << "图像数据大小: " << imageDataSize << std::endl;
        std::cerr << "二进制文件大小: " << fileSize << std::endl;
        file.close();
        return false;
    }
    
    // 分配内存并读取二进制文件内容
    char* binData = new char[fileSize];
    if (!file.read(binData, fileSize)) {
        std::cerr << "读取二进制文件失败!" << std::endl;
        delete[] binData;
        file.close();
        return false;
    }
    file.close();
    
    // 比较图像数据和二进制文件内容
    bool isEqual = memcmp(img.data, binData, fileSize) == 0;
    
    if (isEqual) {
        std::cout << "图像数据与二进制文件内容完全一致！" << std::endl;
    } else {
        std::cout << "图像数据与二进制文件内容不一致！" << std::endl;
        
        // 找出第一个不同的字节位置
        for (size_t i = 0; i < fileSize; ++i) {
            if (reinterpret_cast<unsigned char*>(img.data)[i] != reinterpret_cast<unsigned char*>(binData)[i]) {
                std::cout << "第一个不同字节的位置: " << i << std::endl;
                std::cout << "图像数据中的值: " << static_cast<int>(reinterpret_cast<unsigned char*>(img.data)[i]) << std::endl;
                std::cout << "二进制文件中的值: " << static_cast<int>(reinterpret_cast<unsigned char*>(binData)[i]) << std::endl;
                break;
            }
        }
    }
    
    delete[] binData;
    return isEqual;
}

// 比较模型输出outputs与NPZ文件中的数据是否一致的函数
bool compareOutputsToNpz(rknn_output* outputs, int output_count, const std::string& npzFilePath) {
    try {
        // 加载npz文件
        cnpy::npz_t npz_data = cnpy::npz_load(npzFilePath);
        
        if (npz_data.size() == 0) {
            std::cerr << "NPZ文件为空或无法读取: " << npzFilePath << std::endl;
            return false;
        }
        
        // 获取NPZ文件中的数组数量
        int npz_output_count = npz_data.size();
        std::cout << "NPZ文件中的输出张量数量: " << npz_output_count << ", 实际输出张量数量: " << output_count << std::endl;
        
        if (npz_output_count != output_count) {
            std::cerr << "输出张量数量不匹配!" << std::endl;
            return false;
        }
        
        // 按顺序比较每个输出张量
        for (int i = 0; i < output_count; ++i) {
            // 构造NPZ文件中数组的名称（通常为"arr_0", "arr_1", ... 或其他命名方式）
            std::string arr_name = "arr_" + std::to_string(i);
            
            // 检查是否能找到对应的数组名
            auto it = npz_data.find(arr_name);
            if (it == npz_data.end()) {
                // 如果按默认命名找不到，尝试其他可能的命名方式
                arr_name = "output_" + std::to_string(i);
                it = npz_data.find(arr_name);
                if (it == npz_data.end()) {
                    arr_name = "out" + std::to_string(i);
                    it = npz_data.find(arr_name);
                    if (it == npz_data.end()) {
                        std::cerr << "在NPZ文件中找不到输出数组: " << arr_name << std::endl;
                        continue;
                    }
                }
            }
            
            // 获取NPZ中的数组
            cnpy::NpyArray npz_array = it->second;
            
            // 打印数组信息用于调试
            std::cout << "输出 #" << i << " - NPZ数组形状: ";
            for(size_t j = 0; j < npz_array.shape.size(); j++) {
                std::cout << npz_array.shape[j];
                if(j < npz_array.shape.size()-1) std::cout << "x";
            }
            std::cout << ", 类型: " << npz_array.word_size << " bytes per element" << std::endl;
            
            // 比较大小
            size_t npz_size_bytes = npz_array.num_bytes();
            size_t output_size_bytes = outputs[i].size;
            
            if (npz_size_bytes != output_size_bytes) {
                std::cerr << "输出 #" << i << " 大小不匹配! NPZ: " << npz_size_bytes 
                         << " bytes, 实际输出: " << output_size_bytes << " bytes" << std::endl;
                return false;
            }
            
            // 比较数据内容
            float tolerance = 0.001f; // 允许的误差范围
            bool arrays_equal = true;
            
            // 根据NPZ数组的数据类型进行比较
            if (npz_array.word_size == 4) { // 假设是float类型
                float* npz_data_ptr = reinterpret_cast<float*>(npz_array.data<void>());
                
                // 根据RKNN输出类型决定如何比较
                if (outputs[i].want_float) {
                    float* output_data_ptr = reinterpret_cast<float*>(outputs[i].buf);
                    
                    for (size_t j = 0; j < npz_size_bytes / sizeof(float); ++j) {
                        if (abs(npz_data_ptr[j] - output_data_ptr[j]) > tolerance) {
                            std::cout << "输出 #" << i << ", 第 " << j << " 个元素不匹配: "
                                     << "NPZ值=" << npz_data_ptr[j] 
                                     << ", 实际输出=" << output_data_ptr[j] << std::endl;
                            arrays_equal = false;
                            break;
                        }
                    }
                } else {
                    uint8_t* output_data_ptr = reinterpret_cast<uint8_t*>(outputs[i].buf);
                    
                    for (size_t j = 0; j < npz_size_bytes / sizeof(uint8_t); ++j) {
                        if (abs(npz_data_ptr[j] - static_cast<float>(output_data_ptr[j])) > tolerance) {
                            std::cout << "输出 #" << i << ", 第 " << j << " 个元素不匹配: "
                                     << "NPZ值=" << npz_data_ptr[j] 
                                     << ", 实际输出=" << static_cast<float>(output_data_ptr[j]) << std::endl;
                            arrays_equal = false;
                            break;
                        }
                    }
                }
            } else if (npz_array.word_size == 1) { // 假设是uint8类型
                uint8_t* npz_data_ptr = reinterpret_cast<uint8_t*>(npz_array.data<void>());
                uint8_t* output_data_ptr = reinterpret_cast<uint8_t*>(outputs[i].buf);
                
                for (size_t j = 0; j < npz_size_bytes; ++j) {
                    if (npz_data_ptr[j] != output_data_ptr[j]) {
                        std::cout << "输出 #" << i << ", 第 " << j << " 个元素不匹配: "
                                 << "NPZ值=" << static_cast<int>(npz_data_ptr[j]) 
                                 << ", 实际输出=" << static_cast<int>(output_data_ptr[j]) << std::endl;
                        arrays_equal = false;
                        break;
                    }
                }
            }
            
            if (!arrays_equal) {
                std::cerr << "输出 #" << i << " 数据不匹配!" << std::endl;
                return false;
            } else {
                std::cout << "输出 #" << i << " 数据匹配!" << std::endl;
            }
        }
        
        std::cout << "所有输出张量与NPZ文件中的数据完全匹配！" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "读取NPZ文件时发生错误: " << e.what() << std::endl;
        return false;
    }
}

int main()
{
  std::string model_name = "/home/orangepi/HectorHuang/deploy_percept/runs/models/RK3588/yolov5s_seg.rknn";

  deploy_percept::engine::RknnEngine::Params params;
  params.model_path = model_name;

  deploy_percept::engine::RknnEngine engine(params);

  // 读取图片
  std::string input_path = "/home/orangepi/HectorHuang/deploy_percept/apps/yolov5_seg_rknn/bus.jpg";
  printf("Read %s ...\n", input_path.c_str());
  cv::Mat orig_img = cv::imread(input_path, 1);

  cv::Mat img;
  cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
  deploy_percept::post_process::BoxRect pads;

  cv::Size target_size(engine.model_input_attrs_[0].dims[1], engine.model_input_attrs_[0].dims[1]);
  cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
  // 计算缩放比例
  float scale_w = (float)target_size.width / img.cols;
  float scale_h = (float)target_size.height / img.rows;

  rknn_input inputs[engine.model_io_num_.n_input];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = engine.model_input_attrs_[0].dims[1] * engine.model_input_attrs_[0].dims[2] * engine.model_input_attrs_[0].dims[3];
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;
  inputs[0].buf = img.data;

  // // 比较图像数据与二进制文件内容
  // std::string binFilePath = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_seg/input_image.bin";
  // bool isMatch = compareImageData(img, binFilePath);
  // std::cout << "图像数据与二进制文件匹配结果: " << (isMatch ? "一致" : "不一致") << std::endl;

  rknn_output outputs[engine.model_io_num_.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < engine.model_io_num_.n_output; i++)
  {
    outputs[i].index = i;
    outputs[i].want_float = 0;
  }

  struct timeval start_time, stop_time;
  gettimeofday(&start_time, NULL);
  engine.run(inputs, outputs);
  gettimeofday(&stop_time, NULL);
  printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

  // // 比较模型输出与NPZ文件
  // std::string npzFilePath = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_seg/yolov5_seg_output.npz";
  // bool outputsMatch = compareOutputsToNpz(outputs, engine.model_io_num_.n_output, npzFilePath);
  // std::cout << "模型输出与NPZ文件匹配结果: " << (outputsMatch ? "一致" : "不一致") << std::endl;

  // std::vector<float> out_scales;
  // std::vector<int32_t> out_zps;
  // for (int i = 0; i < engine.model_io_num_.n_output; ++i)
  // {
  //   out_scales.push_back(engine.model_output_attrs_[i].scale);
  //   out_zps.push_back(engine.model_output_attrs_[i].zp);
  // }

  // // 使用YoloV5DetectPostProcess类进行后处理
  // deploy_percept::post_process::YoloV5DetectPostProcess::Params params_post;
  // deploy_percept::post_process::YoloV5DetectPostProcess processor(params_post);

  // processor.run((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf,
  //               target_size.height, target_size.width, pads, scale_w, scale_h, out_zps, out_scales);
  // processor.drawDetectionsResultGroupOnImage(orig_img, processor.getResult().group);

  // std::string out_path = "/home/orangepi/HectorHuang/deploy_percept/tmp/out.jpg";
  // printf("save detect result to %s\n", out_path.c_str());
  // imwrite(out_path, orig_img);

  rknn_outputs_release(engine.ctx_, engine.model_io_num_.n_output, outputs);

  // int test_count = 10;
  // gettimeofday(&start_time, NULL);
  // for (int i = 0; i < test_count; ++i)
  // {
  //   engine.run(inputs, outputs);
  //   rknn_outputs_release(engine.ctx_, engine.model_io_num_.n_output, outputs);
  // }
  // gettimeofday(&stop_time, NULL);
  // printf("loop count = %d , average run  %f ms\n", test_count,
  //        (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

  return 0;
}