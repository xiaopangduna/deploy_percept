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

  

  // rknn_output outputs[engine.model_io_num_.n_output];
  // memset(outputs, 0, sizeof(outputs));
  // for (int i = 0; i < engine.model_io_num_.n_output; i++)
  // {
  //   outputs[i].index = i;
  //   outputs[i].want_float = 0;
  // }

  // struct timeval start_time, stop_time;
  // gettimeofday(&start_time, NULL);
  // engine.run(inputs, outputs);
  // gettimeofday(&stop_time, NULL);
  // printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

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

  // rknn_outputs_release(engine.ctx_, engine.model_io_num_.n_output, outputs);

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