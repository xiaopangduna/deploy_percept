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
#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

#define PROTO_CHANNEL 32
#define PROTO_HEIGHT 160
#define PROTO_WEIGHT 160
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
/**
 * @brief Image rectangle
 *
 */
typedef struct
{
  int left;
  int top;
  int right;
  int bottom;
} image_rect_t;
typedef struct
{
  image_rect_t box;
  float prop;
  int cls_id;
} object_detect_result;

typedef struct
{
  uint8_t *seg_mask;
} object_segment_result;

typedef struct
{
  int id;
  int count;
  object_detect_result results[OBJ_NUMB_MAX_SIZE];
  object_segment_result results_seg[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;
typedef struct
{
  rknn_context rknn_ctx;
  rknn_input_output_num io_num;
  rknn_tensor_attr *input_attrs;
  rknn_tensor_attr *output_attrs;
  int model_channel;
  int model_width;
  int model_height;
  int input_image_width;
  int input_image_height;
  bool is_quant;
} rknn_app_context_t;

typedef struct
{
  int x_pad;
  int y_pad;
  float scale;
} letterbox_t;
// 比较img.data和examples/data/yolov5_seg/input_image.bin内容是否一致的函数
bool compareImageData(const cv::Mat &img, const std::string &binFilePath)
{
  // 读取二进制文件
  std::ifstream file(binFilePath, std::ios::binary | std::ios::ate);
  if (!file.is_open())
  {
    std::cerr << "无法打开二进制文件: " << binFilePath << std::endl;
    return false;
  }

  // 获取文件大小
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // 验证图像数据大小是否匹配
  size_t imageDataSize = img.total() * img.elemSize();
  if (imageDataSize != static_cast<size_t>(fileSize))
  {
    std::cerr << "图像数据大小不匹配!" << std::endl;
    std::cerr << "图像数据大小: " << imageDataSize << std::endl;
    std::cerr << "二进制文件大小: " << fileSize << std::endl;
    file.close();
    return false;
  }

  // 分配内存并读取二进制文件内容
  char *binData = new char[fileSize];
  if (!file.read(binData, fileSize))
  {
    std::cerr << "读取二进制文件失败!" << std::endl;
    delete[] binData;
    file.close();
    return false;
  }
  file.close();

  // 比较图像数据和二进制文件内容
  bool isEqual = memcmp(img.data, binData, fileSize) == 0;

  if (isEqual)
  {
    std::cout << "图像数据与二进制文件内容完全一致！" << std::endl;
  }
  else
  {
    std::cout << "图像数据与二进制文件内容不一致！" << std::endl;

    // 找出第一个不同的字节位置
    for (size_t i = 0; i < fileSize; ++i)
    {
      if (reinterpret_cast<unsigned char *>(img.data)[i] != reinterpret_cast<unsigned char *>(binData)[i])
      {
        std::cout << "第一个不同字节的位置: " << i << std::endl;
        std::cout << "图像数据中的值: " << static_cast<int>(reinterpret_cast<unsigned char *>(img.data)[i]) << std::endl;
        std::cout << "二进制文件中的值: " << static_cast<int>(reinterpret_cast<unsigned char *>(binData)[i]) << std::endl;
        break;
      }
    }
  }

  delete[] binData;
  return isEqual;
}

// 比较模型输出outputs与NPZ文件中的数据是否一致的函数
bool compareOutputsToNpz(rknn_output *outputs, int output_count, const std::string &npzFilePath)
{
  try
  {
    // 加载npz文件
    cnpy::npz_t npz_data = cnpy::npz_load(npzFilePath);

    if (npz_data.size() == 0)
    {
      std::cerr << "NPZ文件为空或无法读取: " << npzFilePath << std::endl;
      return false;
    }

    // 获取NPZ文件中的数组数量
    int npz_output_count = npz_data.size();
    std::cout << "NPZ文件中的输出张量数量: " << npz_output_count << ", 实际输出张量数量: " << output_count << std::endl;

    if (npz_output_count != output_count)
    {
      std::cerr << "输出张量数量不匹配!" << std::endl;
      return false;
    }

    // 按顺序比较每个输出张量
    for (int i = 0; i < output_count; ++i)
    {
      // 构造NPZ文件中数组的名称（通常为"arr_0", "arr_1", ... 或其他命名方式）
      std::string arr_name = "arr_" + std::to_string(i);

      // 检查是否能找到对应的数组名
      auto it = npz_data.find(arr_name);
      if (it == npz_data.end())
      {
        // 如果按默认命名找不到，尝试其他可能的命名方式
        arr_name = "output_" + std::to_string(i);
        it = npz_data.find(arr_name);
        if (it == npz_data.end())
        {
          arr_name = "out" + std::to_string(i);
          it = npz_data.find(arr_name);
          if (it == npz_data.end())
          {
            std::cerr << "在NPZ文件中找不到输出数组: " << arr_name << std::endl;
            continue;
          }
        }
      }

      // 获取NPZ中的数组
      cnpy::NpyArray npz_array = it->second;

      // 打印数组信息用于调试
      std::cout << "输出 #" << i << " - NPZ数组形状: ";
      for (size_t j = 0; j < npz_array.shape.size(); j++)
      {
        std::cout << npz_array.shape[j];
        if (j < npz_array.shape.size() - 1)
          std::cout << "x";
      }
      std::cout << ", 类型: " << npz_array.word_size << " bytes per element" << std::endl;

      // 比较大小
      size_t npz_size_bytes = npz_array.num_bytes();
      size_t output_size_bytes = outputs[i].size;

      if (npz_size_bytes != output_size_bytes)
      {
        std::cerr << "输出 #" << i << " 大小不匹配! NPZ: " << npz_size_bytes
                  << " bytes, 实际输出: " << output_size_bytes << " bytes" << std::endl;
        return false;
      }

      // 比较数据内容
      float tolerance = 0.001f; // 允许的误差范围
      bool arrays_equal = true;

      // 根据NPZ数组的数据类型进行比较
      if (npz_array.word_size == 4)
      { // 假设是float类型
        float *npz_data_ptr = reinterpret_cast<float *>(npz_array.data<void>());

        // 根据RKNN输出类型决定如何比较
        if (outputs[i].want_float)
        {
          float *output_data_ptr = reinterpret_cast<float *>(outputs[i].buf);

          for (size_t j = 0; j < npz_size_bytes / sizeof(float); ++j)
          {
            if (abs(npz_data_ptr[j] - output_data_ptr[j]) > tolerance)
            {
              std::cout << "输出 #" << i << ", 第 " << j << " 个元素不匹配: "
                        << "NPZ值=" << npz_data_ptr[j]
                        << ", 实际输出=" << output_data_ptr[j] << std::endl;
              arrays_equal = false;
              break;
            }
          }
        }
        else
        {
          uint8_t *output_data_ptr = reinterpret_cast<uint8_t *>(outputs[i].buf);

          for (size_t j = 0; j < npz_size_bytes / sizeof(uint8_t); ++j)
          {
            if (abs(npz_data_ptr[j] - static_cast<float>(output_data_ptr[j])) > tolerance)
            {
              std::cout << "输出 #" << i << ", 第 " << j << " 个元素不匹配: "
                        << "NPZ值=" << npz_data_ptr[j]
                        << ", 实际输出=" << static_cast<float>(output_data_ptr[j]) << std::endl;
              arrays_equal = false;
              break;
            }
          }
        }
      }
      else if (npz_array.word_size == 1)
      { // 假设是uint8类型
        uint8_t *npz_data_ptr = reinterpret_cast<uint8_t *>(npz_array.data<void>());
        uint8_t *output_data_ptr = reinterpret_cast<uint8_t *>(outputs[i].buf);

        for (size_t j = 0; j < npz_size_bytes; ++j)
        {
          if (npz_data_ptr[j] != output_data_ptr[j])
          {
            std::cout << "输出 #" << i << ", 第 " << j << " 个元素不匹配: "
                      << "NPZ值=" << static_cast<int>(npz_data_ptr[j])
                      << ", 实际输出=" << static_cast<int>(output_data_ptr[j]) << std::endl;
            arrays_equal = false;
            break;
          }
        }
      }

      if (!arrays_equal)
      {
        std::cerr << "输出 #" << i << " 数据不匹配!" << std::endl;
        return false;
      }
      else
      {
        std::cout << "输出 #" << i << " 数据匹配!" << std::endl;
      }
    }

    std::cout << "所有输出张量与NPZ文件中的数据完全匹配！" << std::endl;
    return true;
  }
  catch (const std::exception &e)
  {
    std::cerr << "读取NPZ文件时发生错误: " << e.what() << std::endl;
    return false;
  }
}
static int read_detection_and_segmentation_results(const char *yaml_path, const char *bin_dir_path, object_detect_result_list *results, int expected_img_width, int expected_img_height)
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
  memset(results, 0, sizeof(object_detect_result_list));

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
      if (current_obj_idx >= OBJ_NUMB_MAX_SIZE)
      {
        printf("Warning: Too many objects in YAML file, skipping rest\n");
        break;
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

int post_process(int model_in_width, int model_in_height, rknn_output *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
  // std::vector<float> filterBoxes;
  // std::vector<float> objProbs;
  // std::vector<int> classId;

  // std::vector<float> filterSegments;
  // float proto[PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT];
  // std::vector<float> filterSegments_by_nms;

  // int validCount = 0;
  // int stride = 0;
  // int grid_h = 0;
  // int grid_w = 0;

  // memset(od_results, 0, sizeof(object_detect_result_list));

  // // process the outputs of rknn
  // for (int i = 0; i < 7; i++)
  // {
  //   grid_h = app_ctx->output_attrs[i].dims[2];
  //   grid_w = app_ctx->output_attrs[i].dims[3];
  //   stride = model_in_height / grid_h;

  //   if (app_ctx->is_quant)
  //   {
  //     validCount += process_i8(outputs, i, (int *)anchor[i / 2], grid_h, grid_w, model_in_height, model_in_width, stride, filterBoxes, filterSegments, proto, objProbs,
  //                              classId, conf_threshold, app_ctx);
  //   }
  //   else
  //   {
  //     validCount += process_fp32(outputs, i, (int *)anchor[i / 2], grid_h, grid_w, model_in_height, model_in_width, stride, filterBoxes, filterSegments, proto, objProbs,
  //                                classId, conf_threshold);
  //   }
  // }

  // // nms
  // if (validCount <= 0)
  // {
  //   return 0;
  // }
  // std::vector<int> indexArray;
  // for (int i = 0; i < validCount; ++i)
  // {
  //   indexArray.push_back(i);
  // }

  // quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  // std::set<int> class_set(std::begin(classId), std::end(classId));

  // for (auto c : class_set)
  // {
  //   nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  // }

  // int last_count = 0;
  // od_results->count = 0;

  // for (int i = 0; i < validCount; ++i)
  // {
  //   if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
  //   {
  //     continue;
  //   }
  //   int n = indexArray[i];

  //   float x1 = filterBoxes[n * 4 + 0];
  //   float y1 = filterBoxes[n * 4 + 1];
  //   float x2 = x1 + filterBoxes[n * 4 + 2];
  //   float y2 = y1 + filterBoxes[n * 4 + 3];
  //   int id = classId[n];
  //   float obj_conf = objProbs[i];

  //   for (int k = 0; k < PROTO_CHANNEL; k++)
  //   {
  //     filterSegments_by_nms.push_back(filterSegments[n * PROTO_CHANNEL + k]);
  //   }

  //   od_results->results[last_count].box.left = x1;
  //   od_results->results[last_count].box.top = y1;
  //   od_results->results[last_count].box.right = x2;
  //   od_results->results[last_count].box.bottom = y2;

  //   od_results->results[last_count].prop = obj_conf;
  //   od_results->results[last_count].cls_id = id;
  //   last_count++;
  // }
  // od_results->count = last_count;
  // int boxes_num = od_results->count;

  // float filterBoxes_by_nms[boxes_num * 4];
  // int cls_id[boxes_num];
  // for (int i = 0; i < boxes_num; i++)
  // {
  //   // for crop_mask
  //   filterBoxes_by_nms[i * 4 + 0] = od_results->results[i].box.left;   // x1;
  //   filterBoxes_by_nms[i * 4 + 1] = od_results->results[i].box.top;    // y1;
  //   filterBoxes_by_nms[i * 4 + 2] = od_results->results[i].box.right;  // x2;
  //   filterBoxes_by_nms[i * 4 + 3] = od_results->results[i].box.bottom; // y2;
  //   cls_id[i] = od_results->results[i].cls_id;

  //   // get real box
  //   od_results->results[i].box.left = box_reverse(od_results->results[i].box.left, model_in_width, letter_box->x_pad, letter_box->scale);
  //   od_results->results[i].box.top = box_reverse(od_results->results[i].box.top, model_in_height, letter_box->y_pad, letter_box->scale);
  //   od_results->results[i].box.right = box_reverse(od_results->results[i].box.right, model_in_width, letter_box->x_pad, letter_box->scale);
  //   od_results->results[i].box.bottom = box_reverse(od_results->results[i].box.bottom, model_in_height, letter_box->y_pad, letter_box->scale);
  // }

  // // compute the mask through Matmul
  // int ROWS_A = boxes_num;
  // int COLS_A = PROTO_CHANNEL;
  // int COLS_B = PROTO_HEIGHT * PROTO_WEIGHT;
  // uint8_t *matmul_out = (uint8_t *)malloc(boxes_num * PROTO_HEIGHT * PROTO_WEIGHT * sizeof(uint8_t));
  // matmul_by_cpu_uint8(filterSegments_by_nms, proto, matmul_out, ROWS_A, COLS_A, COLS_B);

  // uint8_t *seg_mask = (uint8_t *)malloc(boxes_num * model_in_height * model_in_width * sizeof(uint8_t));
  // resize_by_opencv_uint8(matmul_out, PROTO_WEIGHT, PROTO_HEIGHT, boxes_num, seg_mask, model_in_width, model_in_height);

  // // crop mask
  // uint8_t *all_mask_in_one = (uint8_t *)malloc(model_in_height * model_in_width * sizeof(uint8_t));
  // memset(all_mask_in_one, 0, model_in_height * model_in_width * sizeof(uint8_t));
  // crop_mask_uint8(seg_mask, all_mask_in_one, filterBoxes_by_nms, boxes_num, cls_id, model_in_height, model_in_width);

  // // get real mask
  // int cropped_height = model_in_height - letter_box->y_pad * 2;
  // int cropped_width = model_in_width - letter_box->x_pad * 2;
  // int ori_in_height = app_ctx->input_image_height;
  // int ori_in_width = app_ctx->input_image_width;
  // int y_pad = letter_box->y_pad;
  // int x_pad = letter_box->x_pad;
  // uint8_t *cropped_seg_mask = (uint8_t *)malloc(cropped_height * cropped_width * sizeof(uint8_t));
  // uint8_t *real_seg_mask = (uint8_t *)malloc(ori_in_height * ori_in_width * sizeof(uint8_t));
  // seg_reverse(all_mask_in_one, cropped_seg_mask, real_seg_mask,
  //             model_in_height, model_in_width, cropped_height, cropped_width, ori_in_height, ori_in_width, y_pad, x_pad);
  // od_results->results_seg[0].seg_mask = real_seg_mask;
  // free(all_mask_in_one);
  // free(cropped_seg_mask);
  // free(seg_mask);
  // free(matmul_out);

  return 0;
};

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

  // 读取保存的检测和分割结果用于验证
  std::string yaml_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_seg/detection_results.yaml";
  std::string bin_dir_path = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_seg";
  int img_width = 640;  // 根据实际情况设置
  int img_height = 640; // 根据实际情况设置

  object_detect_result_list loaded_results;
  if (read_detection_and_segmentation_results(yaml_path.c_str(), bin_dir_path.c_str(), &loaded_results, img_width, img_height) != 0)
  {
    printf("Failed to read saved data for verification\n");
    return -1;
  }

  printf("Successfully loaded %d objects from YAML file\n", loaded_results.count);

  rknn_app_context_t rknn_app_ctx;
  memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
  int model_in_width;
  int model_in_height;
  model_in_height = engine.model_input_attrs_[0].dims[1];
  model_in_width = engine.model_input_attrs_[0].dims[0];
  letterbox_t letter_box;
  memset(&letter_box, 0, sizeof(letterbox_t));
  letter_box.scale = 1.0;
  letter_box.x_pad = 0;
  letter_box.y_pad = 0;
  const float nms_threshold = NMS_THRESH;
  const float box_conf_threshold = BOX_THRESH;
  object_detect_result_list od_results;
  memset(&od_results, 0x00, sizeof(od_results));
  post_process(model_in_width, model_in_height, outputs, &letter_box, box_conf_threshold, nms_threshold, &od_results);

  // std::vector<float> out_scales;
  // std::vector<int32_t> out_zps;
  // for (int i = 0; i < engine.model_io_num_.n_output; ++i)
  // {
  //   out_scales.push_back(engine.model_output_attrs_[i].scale);
  //   out_zps.push_back(engine.model_output_attrs_[i].zp);
  // }

  // 使用YoloV5DetectPostProcess类进行后处理
  // deploy_percept::post_process::YoloV5DetectPostProcess::Params params_post;
  // deploy_percept::post_process::YoloV5DetectPostProcess processor(params_post);

  // processor.run((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf,
  //               target_size.height, target_size.width, pads, scale_w, scale_h, out_zps, out_scales);
  // processor.drawDetectionsResultGroupOnImage(orig_img, processor.getResult().group);

  // 绘制从YAML文件加载的检测和分割结果到图像上
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

  // 首先绘制分割掩码
  if (loaded_results.count >= 1)
  {
    int width = orig_img.cols;
    int height = orig_img.rows;
    float alpha = 0.5f; // 透明度

    for (int i = 0; i < loaded_results.count; i++)
    {
      object_detect_result *det_result = &(loaded_results.results[i]);

      // 获取对应类别的颜色
      cv::Vec3b color = cv::Vec3b(class_colors[det_result->cls_id % 20][0],
                                  class_colors[det_result->cls_id % 20][1],
                                  class_colors[det_result->cls_id % 20][2]); // RGB格式

      // 绘制分割掩码
      if (loaded_results.results_seg[i].seg_mask != nullptr)
      {
        // 直接修改原图的像素值
        for (int h = 0; h < height; h++)
        {
          for (int w = 0; w < width; w++)
          {
            // 获取掩码值，这个值可能代表类别或实例ID
            int mask_value = loaded_results.results_seg[i].seg_mask[h * width + w];

            if (mask_value != 0)
            {
              // 使用掩码值来索引颜色，而不是使用检测框的类别ID
              cv::Vec3b color = cv::Vec3b(class_colors[mask_value % 20][0],
                                          class_colors[mask_value % 20][1],
                                          class_colors[mask_value % 20][2]); // RGB格式

              cv::Vec3b &pixel = orig_img.at<cv::Vec3b>(h, w);

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
    for (int i = 0; i < loaded_results.count; i++)
    {
      object_detect_result *det_result = &(loaded_results.results[i]);

      // 获取对应类别的颜色
      cv::Scalar color = cv::Scalar(class_colors[det_result->cls_id % 20][2],
                                    class_colors[det_result->cls_id % 20][1],
                                    class_colors[det_result->cls_id % 20][0]); // BGR格式

      // 绘制边界框
      cv::rectangle(orig_img,
                    cv::Point(det_result->box.left, det_result->box.top),
                    cv::Point(det_result->box.right, det_result->box.bottom),
                    color, 2);

      // 添加标签文本
      std::string label = "Class " + std::to_string(det_result->cls_id) + " " +
                          std::to_string(det_result->prop * 100) + "%";
      int baseline;
      cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
      cv::rectangle(orig_img,
                    cv::Point(det_result->box.left, det_result->box.top - textSize.height - 10),
                    cv::Point(det_result->box.left + textSize.width, det_result->box.top),
                    color, -1);
      cv::putText(orig_img, label,
                  cv::Point(det_result->box.left, det_result->box.top - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
  }

  std::string out_path = "/home/orangepi/HectorHuang/deploy_percept/tmp/out.jpg";
  printf("save detect result to %s\n", out_path.c_str());
  cv::imwrite(out_path, orig_img);

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