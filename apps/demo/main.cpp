#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "rknn_api.h"

#include "cnpy.h"
#include <stdio.h>
// #include "im2d.h"
// #include "rga.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <sys/time.h>
#include <fstream>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <vector>

// 添加deploy_percept相关头文件
#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include "deploy_percept/engine/BaseEngine.hpp" // 添加BaseEngine头文件
#include "deploy_percept/engine/RknnEngine.hpp" // 添加RknnEngine头文件

// #define PERF_WITH_POST 1
// #define OBJ_NAME_MAX_SIZE 16
// #define OBJ_NUMB_MAX_SIZE 64
// #define OBJ_CLASS_NUM 80

// #define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

#define LABEL_NALE_TXT_PATH "/home/orangepi/HectorHuang/deploy_percept/apps/demo/coco_80_labels_list.txt"

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }


int main()
{
  const char *model_name = "/home/orangepi/HectorHuang/deploy_percept/runs/models/RK3588/yolov5s-640-640.rknn";

  deploy_percept::engine::RknnEngine::Params params;
  params.model_path = std::string(model_name);
  deploy_percept::engine::RknnEngine engine(params);

  rknn_context ctx = engine.ctx_;

  rknn_input_output_num io_num;
  io_num = engine.model_io_num_;
  int ret;

  int channel = 3;
  int width = 0;
  int height = 0;
  if (engine.model_input_attrs_[0].fmt == RKNN_TENSOR_NCHW)
  {
    printf("model is NCHW input fmt\n");
    channel = engine.model_input_attrs_[0].dims[1];
    height = engine.model_input_attrs_[0].dims[2];
    width = engine.model_input_attrs_[0].dims[3];
  }
  else
  {
    printf("model is NHWC input fmt\n");
    height = engine.model_input_attrs_[0].dims[1];
    width = engine.model_input_attrs_[0].dims[2];
    channel = engine.model_input_attrs_[0].dims[3];
  }
  printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = engine.model_input_attrs_[0].dims[1] * engine.model_input_attrs_[0].dims[2] * engine.model_input_attrs_[0].dims[3];
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;
  // 读取图片
  std::string input_path = "/home/orangepi/HectorHuang/deploy_percept/apps/demo/bus.jpg";
  printf("Read %s ...\n", input_path.c_str());
  cv::Mat orig_img = cv::imread(input_path, 1);
  if (!orig_img.data)
  {
    printf("cv::imread %s fail!\n", input_path.c_str());
    return -1;
  }
  cv::Mat img;
  cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
  int img_width = 0;
  int img_height = 0;
  img_width = img.cols;
  img_height = img.rows;
  printf("img width = %d, img height = %d\n", img_width, img_height);

  // 指定目标大小和预处理方式,默认使用LetterBox的预处理
  deploy_percept::post_process::BoxRect pads;
  // memset(&pads, 0, sizeof(deploy_percept::post_process::BoxRect));
  cv::Size target_size(width, height);
  cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
  // 计算缩放比例
  float scale_w = (float)target_size.width / img.cols;
  float scale_h = (float)target_size.height / img.rows;

  inputs[0].buf = img.data;

  struct timeval start_time, stop_time;
  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < io_num.n_output; i++)
  {
    outputs[i].index = i;
    outputs[i].want_float = 0;
  }

  gettimeofday(&start_time, NULL);
  engine.run(inputs, outputs);
  gettimeofday(&stop_time, NULL);

  printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

  // 后处理 - 使用YoloV5DetectPostProcess类
  std::vector<float> out_scales;
  std::vector<int32_t> out_zps;
  for (int i = 0; i < io_num.n_output; ++i)
  {
    out_scales.push_back(engine.model_output_attrs_[i].scale);
    out_zps.push_back(engine.model_output_attrs_[i].zp);
  }

  // 使用YoloV5DetectPostProcess类进行后处理
  deploy_percept::post_process::YoloV5DetectPostProcess::Params params_post;
  deploy_percept::post_process::YoloV5DetectPostProcess processor(params_post);

  processor.run((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf,
                height, width, pads, scale_w, scale_h, out_zps, out_scales);
  processor.drawDetectionsResultGroupOnImage(orig_img, processor.getResult().group);

  std::string out_path = "/home/orangepi/HectorHuang/deploy_percept/tmp/out.jpg";
  printf("save detect result to %s\n", out_path.c_str());
  imwrite(out_path, orig_img);

  ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

  // 耗时统计
  int test_count = 10;
  gettimeofday(&start_time, NULL);
  for (int i = 0; i < test_count; ++i)
  {
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
  }
  gettimeofday(&stop_time, NULL);
  printf("loop count = %d , average run  %f ms\n", test_count,
         (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

  return 0;
}