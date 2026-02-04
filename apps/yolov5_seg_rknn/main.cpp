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
#include "deploy_percept/post_process/YoloV5SegPostProcess.hpp" // 新增头文件
#include "deploy_percept/post_process/types.hpp"
#include "deploy_percept/engine/RknnEngine.hpp"

#define OBJ_NUMB_MAX_SIZE 128

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

/**
 * @brief Image rectangle
 *
 */
// 使用统一的BoxRect结构体
// typedef struct
// {
//   int left;
//   int top;
//   int right;
//   int bottom;
// } image_rect_t;

// 直接使用统一的结构体，不再需要自定义的object_detect_result_list

// 打印结果详情的函数
void printResultDetails(const deploy_percept::post_process::ResultGroup &result, const std::string &label)
{
  printf("\n=== %s ===\n", label.c_str());
  printf("Count: %d\n", result.count);

  for (int i = 0; i < result.count; i++)
  {
    printf("Object %d:\n", i);
    printf("  BBox: (%d, %d, %d, %d)\n",
           result.results[i].box.left,
           result.results[i].box.top,
           result.results[i].box.right,
           result.results[i].box.bottom);
    printf("  Prop: %.8f\n", result.results[i].prop);
    printf("  Class ID: %d\n", result.results[i].cls_id);

    if (result.results_seg.size() > i && result.results_seg[i].seg_mask != nullptr)
    {
      printf("  Mask: exists\n");
      // 打印前几个掩码值用于调试
      printf("  First 10 mask values: ");
      for (int j = 0; j < 10 && j < 640 * 640; j++)
      {
        printf("%d ", result.results_seg[i].seg_mask[j]);
      }
      printf("\n");
    }
    else
    {
      printf("  Mask: null\n");
    }
  }
}

// 比较两个检测结果列表是否一致
bool compareDetectionResults(const deploy_percept::post_process::ResultGroup &loaded, const deploy_percept::post_process::ResultGroup &computed, int img_width, int img_height)
{
  if (loaded.count != computed.count)
  {
    printf("Object counts differ: loaded=%d, computed=%d\n", loaded.count, computed.count);
    return false;
  }

  printf("Comparing %d objects...\n", loaded.count);

  for (int i = 0; i < loaded.count; i++)
  {
    const deploy_percept::post_process::DetectResult &loaded_result = loaded.results[i];
    const deploy_percept::post_process::DetectResult &computed_result = computed.results[i];

    // 比较边界框
    if (loaded_result.box.left != computed_result.box.left ||
        loaded_result.box.top != computed_result.box.top ||
        loaded_result.box.right != computed_result.box.right ||
        loaded_result.box.bottom != computed_result.box.bottom)
    {
      printf("Bounding box mismatch for object %d: loaded=(%d,%d,%d,%d), computed=(%d,%d,%d,%d)\n",
             i,
             loaded_result.box.left, loaded_result.box.top, loaded_result.box.right, loaded_result.box.bottom,
             computed_result.box.left, computed_result.box.top, computed_result.box.right, computed_result.box.bottom);
      return false;
    }

    // 比较置信度（允许一定误差）
    if (abs(loaded_result.prop - computed_result.prop) > 0.001f)
    {
      printf("Confidence mismatch for object %d: loaded=%.8f, computed=%.8f\n",
             i, loaded_result.prop, computed_result.prop);
      return false;
    }

    // 比较类别ID
    if (loaded_result.cls_id != computed_result.cls_id)
    {
      printf("Class ID mismatch for object %d: loaded=%d, computed=%d\n",
             i, loaded_result.cls_id, computed_result.cls_id);
      return false;
    }
  }

  // 比较分割掩码 - 所有掩码都存储在results_seg[0]中
  if (loaded.results_seg.size() > 0 && computed.results_seg.size() > 0 &&
      loaded.results_seg[0].seg_mask != nullptr && computed.results_seg[0].seg_mask != nullptr)
  {
    // 比较整个掩码
    bool masks_match = true;
    int total_pixels = img_width * img_height;

    // 统计不匹配的像素数量
    int mismatch_count = 0;
    int first_diff_idx = -1; // 记录第一个不同像素的位置

    for (int j = 0; j < total_pixels; j++)
    {
      if (loaded.results_seg[0].seg_mask[j] != computed.results_seg[0].seg_mask[j])
      {
        if (first_diff_idx == -1)
        {
          first_diff_idx = j;
        }
        mismatch_count++;
      }
    }

    if (mismatch_count > 0)
    {
      printf("Segmentation mask mismatch: %d/%d pixels differ\n", mismatch_count, total_pixels);
      printf("First segmentation mask mismatch at pixel index %d\n", first_diff_idx);
      printf("Loaded mask value at first diff index: %d\n", loaded.results_seg[0].seg_mask[first_diff_idx]);
      printf("Computed mask value at first diff index: %d\n", computed.results_seg[0].seg_mask[first_diff_idx]);

      // 输出更多上下文信息
      int row = first_diff_idx / img_width;
      int col = first_diff_idx % img_width;
      printf("Pixel coordinates (row, col) of first mismatch: (%d, %d)\n", row, col);

      // 检查附近像素的值
      printf("Nearby pixels in loaded mask: ");
      for (int k = std::max(0, first_diff_idx - 3); k <= std::min(total_pixels - 1, first_diff_idx + 3); k++)
      {
        printf("[%d]%d ", k, loaded.results_seg[0].seg_mask[k]);
      }
      printf("\n");

      printf("Nearby pixels in computed mask: ");
      for (int k = std::max(0, first_diff_idx - 3); k <= std::min(total_pixels - 1, first_diff_idx + 3); k++)
      {
        printf("[%d]%d ", k, computed.results_seg[0].seg_mask[k]);
      }
      printf("\n");

      // 分析每个对象在该位置是否有mask
      printf("Analyzing mask contribution at pixel (%d, %d):\n", row, col);
      for (int i = 0; i < loaded.count; i++)
      {
        int x = col;
        int y = row;
        int pixel_idx = y * img_width + x;

        // Check if this pixel is within the bounding box of each object
        if (x >= loaded.results[i].box.left && x <= loaded.results[i].box.right &&
            y >= loaded.results[i].box.top && y <= loaded.results[i].box.bottom)
        {
          printf("  Object %d (bbox: %d-%d, %d-%d) covers this pixel\n",
                 i,
                 loaded.results[i].box.left, loaded.results[i].box.right,
                 loaded.results[i].box.top, loaded.results[i].box.bottom);
        }
      }

      return false;
    }
    else
    {
      printf("All segmentation mask pixels match!\n");
    }
  }
  else if ((loaded.results_seg.size() > 0 && loaded.results_seg[0].seg_mask != nullptr) != 
           (computed.results_seg.size() > 0 && computed.results_seg[0].seg_mask != nullptr))
  {
    printf("One segmentation mask exists but the other doesn't\n");
    if (loaded.results_seg.size() > 0 && loaded.results_seg[0].seg_mask != nullptr)
    {
      printf("Loaded has mask, computed does not\n");
    }
    else
    {
      printf("Computed has mask, loaded does not\n");
    }
    return false;
  }

  printf("All results match!\n");
  return true;
}

// 绘制检测和分割结果
void drawDetectionResults(cv::Mat &image, const deploy_percept::post_process::ResultGroup &results)
{
  // 定义类别颜色
  unsigned char class_colors[][3] = {
      {255, 56, 56},   // 'FF3838'
      {255, 157, 151}, // 'FF9D97'
      {255, 112, 31},  // 'FF701F'
      {255, 178, 29},  // 'FFB21D'
      {207, 210, 49},  // 'CFD231'
      {72, 249, 10},   // '48F90A'
      {146, 204, 23},  // '92CC17'
      {61, 219, 134},  // '3DDB86'
      {26, 147, 52},   // '1A9334'
      {0, 212, 187},   // '00D4BB'
      {44, 153, 168},  // '2C99A8'
      {0, 194, 255},   // '00C2FF'
      {52, 69, 147},   // '344593'
      {100, 115, 255}, // '6473FF'
      {0, 24, 236},    // '0018EC'
      {132, 56, 255},  // '8438FF'
      {82, 0, 133},    // '520085'
      {203, 56, 255},  // 'CB38FF'
      {255, 149, 200}, // 'FF95C8'
      {255, 55, 199}   // 'FF37C7'
  };

  int width = image.cols;
  int height = image.rows;
  float alpha = 0.5f; // 透明度

  // 首先绘制分割掩码
  if (results.count >= 1)
  {
    for (int i = 0; i < results.count; i++)
    {
      const deploy_percept::post_process::DetectResult *det_result = &results.results[i];

      // 获取对应类别的颜色
      cv::Vec3b color = cv::Vec3b(class_colors[det_result->cls_id % 20][0],
                                  class_colors[det_result->cls_id % 20][1],
                                  class_colors[det_result->cls_id % 20][2]); // RGB格式

      // 绘制分割掩码
      if (results.results_seg.size() > i && results.results_seg[i].seg_mask != nullptr)
      {
        // 直接修改原图的像素值
        for (int h = 0; h < height; h++)
        {
          for (int w = 0; w < width; w++)
          {
            // 获取掩码值
            int mask_value = results.results_seg[i].seg_mask[h * width + w];

            if (mask_value != 0)
            {
              // 使用掩码值来索引颜色
              cv::Vec3b color = cv::Vec3b(class_colors[mask_value % 20][0],
                                          class_colors[mask_value % 20][1],
                                          class_colors[mask_value % 20][2]); // RGB格式

              cv::Vec3b &pixel = image.at<cv::Vec3b>(h, w);

              // 使用对象的类别颜色来绘制掩码
              pixel[0] = (unsigned char)(color[0] * (1 - alpha) + pixel[0] * alpha); // B
              pixel[1] = (unsigned char)(color[1] * (1 - alpha) + pixel[1] * alpha); // G
              pixel[2] = (unsigned char)(color[2] * (1 - alpha) + pixel[2] * alpha); // R
            }
          }
        }
      }
    }

    // 然后绘制边界框和标签
    for (int i = 0; i < results.count; i++)
    {
      const deploy_percept::post_process::DetectResult *det_result = &results.results[i];

      // 获取对应类别的颜色
      cv::Scalar color = cv::Scalar(class_colors[det_result->cls_id % 20][2],
                                    class_colors[det_result->cls_id % 20][1],
                                    class_colors[det_result->cls_id % 20][0]); // BGR格式

      // 绘制边界框
      cv::rectangle(image,
                    cv::Point(det_result->box.left, det_result->box.top),
                    cv::Point(det_result->box.right, det_result->box.bottom),
                    color, 2);

      // 添加标签文本
      std::string label = "Class " + std::to_string(det_result->cls_id) + " " +
                          std::to_string(det_result->prop * 100) + "%";
      int baseline;
      cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
      cv::rectangle(image,
                    cv::Point(det_result->box.left, det_result->box.top - textSize.height - 10),
                    cv::Point(det_result->box.left + textSize.width, det_result->box.top),
                    color, -1);
      cv::putText(image, label,
                  cv::Point(det_result->box.left, det_result->box.top - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
  }
}

static int read_detection_and_segmentation_results(const char *yaml_path, const char *bin_dir_path, deploy_percept::post_process::ResultGroup *results, int expected_img_width, int expected_img_height)
{
  if (!yaml_path || !bin_dir_path || !results)
  {
    printf("Invalid parameters for read_detection_and_segmentation_results\n");
    return -1;
  }

  FILE *fp = fopen(yaml_path, "r");
  if (!fp)
  {
    printf("Failed to open YAML file for reading: %s\n", yaml_path);
    return -1;
  }

  // 初始化结果结构
  results->id = 0;
  results->count = 0;
  results->results.clear();
  results->results_seg.clear();

  // 读取文件内容并解析
  char line[1024];
  int current_obj_idx = -1;
  int obj_count = 0;

  while (fgets(line, sizeof(line), fp))
  {
    // 检查是否是新对象的开始
    if (strstr(line, "object_"))
    {
      current_obj_idx++;
      obj_count++;
      
      // 确保vector有足够的空间
      if (results->results.size() <= current_obj_idx)
      {
        results->results.resize(current_obj_idx + 1);
      }
      
      if (results->results_seg.size() <= current_obj_idx)
      {
        results->results_seg.resize(current_obj_idx + 1);
      }
    }
    // 解析边界框信息
    else if (current_obj_idx >= 0 && strstr(line, "left:"))
    {
      sscanf(line, "          left: %d", &results->results[current_obj_idx].box.left);
    }
    else if (current_obj_idx >= 0 && strstr(line, "top:"))
    {
      sscanf(line, "          top: %d", &results->results[current_obj_idx].box.top);
    }
    else if (current_obj_idx >= 0 && strstr(line, "right:"))
    {
      sscanf(line, "          right: %d", &results->results[current_obj_idx].box.right);
    }
    else if (current_obj_idx >= 0 && strstr(line, "bottom:"))
    {
      sscanf(line, "          bottom: %d", &results->results[current_obj_idx].box.bottom);
    }
    // 解析置信度
    else if (current_obj_idx >= 0 && strstr(line, "confidence:"))
    {
      double conf;
      sscanf(line, "        confidence: %lf", &conf);
      results->results[current_obj_idx].prop = (float)conf;
    }
    // 解析类别ID
    else if (current_obj_idx >= 0 && strstr(line, "class_id:"))
    {
      sscanf(line, "        class_id: %d", &results->results[current_obj_idx].cls_id);
      // 设置name字段
      snprintf(results->results[current_obj_idx].name, sizeof(results->results[current_obj_idx].name), 
               "class_%d", results->results[current_obj_idx].cls_id);
    }
    // 解析分割掩码文件路径
    else if (current_obj_idx >= 0 && strstr(line, "segmentation_mask_file:"))
    {
      char mask_filename[256];
      if (sscanf(line, "        segmentation_mask_file: \"%255[^\"]\"", mask_filename) == 1)
      {
        if (strcmp(mask_filename, "none") != 0 && strcmp(mask_filename, "failed_to_save") != 0)
        {
          // 构建完整路径
          char full_path[512];
          snprintf(full_path, sizeof(full_path), "%s/%s", bin_dir_path, mask_filename);

          // 读取二进制掩码文件
          FILE *mask_fp = fopen(full_path, "rb");
          if (mask_fp)
          {
            // 获取文件大小
            fseek(mask_fp, 0, SEEK_END);
            long mask_size = ftell(mask_fp);
            fseek(mask_fp, 0, SEEK_SET);

            // 分配内存并读取数据
            results->results_seg[current_obj_idx].seg_mask = (uint8_t *)malloc(mask_size);
            if (results->results_seg[current_obj_idx].seg_mask)
            {
              size_t read_size = fread(results->results_seg[current_obj_idx].seg_mask, 1, mask_size, mask_fp);
              if (read_size != mask_size)
              {
                printf("Warning: Could not read full mask data for object %d\n", current_obj_idx);
                free(results->results_seg[current_obj_idx].seg_mask);
                results->results_seg[current_obj_idx].seg_mask = NULL;
              }
            }
            else
            {
              printf("Failed to allocate memory for mask of object %d\n", current_obj_idx);
            }

            fclose(mask_fp);
          }
          else
          {
            printf("Could not open mask file: %s\n", full_path);
          }
        }
      }
    }
  }

  results->id = 0; // 设置默认ID
  results->count = obj_count;

  fclose(fp);

  printf("Successfully read detection and segmentation results from YAML: %s\n", yaml_path);
  printf("Parsed %d objects\n", obj_count);

  return 0;
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

  // 读取保存的检测和分割结果用于验证
  std::string yaml_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_seg/detection_results.yaml";
  std::string bin_dir_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_seg";
  int img_width = 640;  // 根据实际情况设置
  int img_height = 640; // 根据实际情况设置

  deploy_percept::post_process::ResultGroup loaded_results;
  if (read_detection_and_segmentation_results(yaml_path.c_str(), bin_dir_path.c_str(), &loaded_results, img_width, img_height) != 0)
  {
    printf("Failed to read saved data for verification\n");
    return -1;
  }

  printf("Successfully loaded %d objects from YAML file\n", loaded_results.count);

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

  // 打印loaded_results和seg_results的详细信息
  printResultDetails(loaded_results, "LOADED RESULTS");
  printResultDetails(seg_results, "COMPUTED RESULTS");

  // 比较加载的结果和计算的结果
  printf("Comparing loaded results with computed results...\n");
  bool results_match = compareDetectionResults(loaded_results, seg_results, orig_img.cols, orig_img.rows);
  printf("Results match: %s\n", results_match ? "YES" : "NO");

  // 绘制计算得到的检测结果
  cv::Mat result_img = orig_img.clone();
  drawDetectionResults(result_img, seg_results);

  std::string computed_out_path = "/home/orangepi/HectorHuang/deploy_percept/tmp/computed_out.jpg";
  printf("Save computed detect result to %s\n", computed_out_path.c_str());
  cv::imwrite(computed_out_path, result_img);

  cv::Mat result_img_2 = orig_img.clone();
  drawDetectionResults(result_img_2, loaded_results);
  std::string computed_out_path_2 = "/home/orangepi/HectorHuang/deploy_percept/tmp/out.jpg";
  printf("Save loaded detect result to %s\n", computed_out_path_2.c_str());
  cv::imwrite(computed_out_path_2, result_img_2);

  rknn_outputs_release(engine.ctx_, engine.model_io_num_.n_output, outputs);

  // 释放分配的掩码内存
  for (size_t i = 0; i < loaded_results.results_seg.size(); ++i)
  {
    if (loaded_results.results_seg[i].seg_mask != nullptr)
    {
      free(loaded_results.results_seg[i].seg_mask);
      loaded_results.results_seg[i].seg_mask = nullptr;
    }
  }

  return 0;
}