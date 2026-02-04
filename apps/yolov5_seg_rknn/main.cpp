#include <iostream>
#include <stdio.h>
#include <string>
#include <memory>
#include <vector>
#include <sys/time.h>
#include <chrono>

#include "rknn_api.h"

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

// #include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/post_process/YoloV5SegPostProcess.hpp" // 新增头文件
#include "deploy_percept/post_process/types.hpp"
#include "deploy_percept/engine/RknnEngine.hpp"

#define OBJ_NUMB_MAX_SIZE 128

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

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

  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(engine.model_input_attrs_[0].dims[2], engine.model_input_attrs_[0].dims[1]));

  rknn_input inputs[engine.model_io_num_.n_input];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = engine.model_input_attrs_[0].dims[1] * engine.model_input_attrs_[0].dims[2] * engine.model_input_attrs_[0].dims[3];
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;
  inputs[0].buf = resized_img.data;

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

  // 为兼容新的post_process函数接口，创建std::vector<void*>
  std::vector<void *> output_buffers;
  for (int i = 0; i < engine.model_io_num_.n_output; i++)
  {
    output_buffers.push_back(outputs[i].buf);
  }

  // 将rknn_tensor_attr转换为基本数据类型
  std::vector<std::vector<int>> output_dims;
  std::vector<float> output_scales;
  std::vector<int32_t> output_zps;

  for (int i = 0; i < engine.model_io_num_.n_output; i++)
  {
    std::vector<int> dims(4);
    dims[0] = engine.model_output_attrs_[i].dims[0];
    dims[1] = engine.model_output_attrs_[i].dims[1];
    dims[2] = engine.model_output_attrs_[i].dims[2];
    dims[3] = engine.model_output_attrs_[i].dims[3];
    output_dims.push_back(dims);
    output_scales.push_back(engine.model_output_attrs_[i].scale);
    output_zps.push_back(engine.model_output_attrs_[i].zp);
  }

  // 使用YoloV5SegPostProcess类进行后处理
  deploy_percept::post_process::YoloV5SegPostProcess::Params params_post;
  deploy_percept::post_process::YoloV5SegPostProcess seg_processor(params_post);

  // // 调用新的后处理类
  bool success = seg_processor.run(
      output_dims,
      output_scales,
      output_zps,
      &output_buffers,
      orig_img.rows,
      orig_img.cols);

  if (!success)
  {
    printf("Post-processing failed: %s\n", seg_processor.getResult().message.c_str());
    return -1;
  }

  // 获取后处理结果
  auto seg_results = seg_processor.getResult().group;



  // 绘制计算得到的检测结果 - 使用类的成员函数
  cv::Mat result_img = orig_img.clone();
  seg_processor.drawDetectionResults(result_img, seg_results);

  std::string computed_out_path = "/home/orangepi/HectorHuang/deploy_percept/tmp/computed_out.jpg";
  printf("Save computed detect result to %s\n", computed_out_path.c_str());
  cv::imwrite(computed_out_path, result_img);

  rknn_outputs_release(engine.ctx_, engine.model_io_num_.n_output, outputs);

  return 0;
}