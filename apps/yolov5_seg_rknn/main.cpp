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
 * @brief 通用张量属性结构体
 */
typedef struct
{
  int dims[4]; // 张量维度
  float scale; // 量化参数scale
  int32_t zp;  // 量化参数zero point
} TensorAttr;

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

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
  float key;
  int key_index;
  int low = left;
  int high = right;
  if (left < right)
  {
    key_index = indices[left];
    key = input[left];
    while (low < high)
    {
      while (low < high && input[high] <= key)
      {
        high--;
      }
      input[low] = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key)
      {
        low++;
      }
      input[high] = input[low];
      indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

// 比较两个检测结果列表是否一致
bool compareDetectionResults(const object_detect_result_list &loaded, const object_detect_result_list &computed, int img_width, int img_height)
{
  if (loaded.count != computed.count)
  {
    printf("Object counts differ: loaded=%d, computed=%d\n", loaded.count, computed.count);
    return false;
  }

  printf("Comparing %d objects...\n", loaded.count);

  for (int i = 0; i < loaded.count; i++)
  {
    const object_detect_result &loaded_result = loaded.results[i];
    const object_detect_result &computed_result = computed.results[i];

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
    if (abs(loaded_result.prop - computed_result.prop) > 0.01f)
    {
      printf("Confidence mismatch for object %d: loaded=%.4f, computed=%.4f\n",
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

    // 比较分割掩码（如果存在）
    if (loaded.results_seg[i].seg_mask != nullptr && computed.results_seg[i].seg_mask != nullptr)
    {
      // 比较整个掩码
      bool masks_match = true;
      int total_pixels = img_width * img_height;
      for (int j = 0; j < total_pixels; j++)
      {
        if (loaded.results_seg[i].seg_mask[j] != computed.results_seg[i].seg_mask[j])
        {
          masks_match = false;
          break;
        }
      }
      if (!masks_match)
      {
        printf("Segmentation mask mismatch for object %d\n", i);
        return false;
      }
    }
    else if ((loaded.results_seg[i].seg_mask != nullptr) != (computed.results_seg[i].seg_mask != nullptr))
    {
      printf("One segmentation mask exists but the other doesn't for object %d\n", i);
      return false;
    }
  }

  printf("All results match!\n");
  return true;
}

// 绘制检测和分割结果
// 绘制检测和分割结果
void drawDetectionResults(cv::Mat &image, const object_detect_result_list &results)
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
      object_detect_result *det_result = (object_detect_result *)&results.results[i];

      // 获取对应类别的颜色
      cv::Vec3b color = cv::Vec3b(class_colors[det_result->cls_id % 20][0],
                                  class_colors[det_result->cls_id % 20][1],
                                  class_colors[det_result->cls_id % 20][2]); // RGB格式

      // 绘制分割掩码
      if (results.results_seg[i].seg_mask != nullptr)
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
      object_detect_result *det_result = (object_detect_result *)&results.results[i];

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
const int anchor[3][6] = {{10, 13, 16, 30, 33, 23},
                          {30, 61, 62, 45, 59, 119},
                          {116, 90, 156, 198, 373, 326}};

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }
inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}
static int process_i8(std::vector<void *> *all_input, int input_id, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                      std::vector<float> &boxes, std::vector<float> &segments, float *proto, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                      std::vector<std::vector<int>> &output_dims, std::vector<float> &output_scales, std::vector<int32_t> &output_zps)
{

  int validCount = 0;
  int grid_len = grid_h * grid_w;

  if (input_id % 2 == 1)
  {
    return validCount;
  }

  if (input_id == 6)
  {
    int8_t *input_proto = (int8_t *)(*all_input)[input_id];
    int32_t zp_proto = output_zps[input_id];
    float scale_proto = output_scales[input_id];
    for (int i = 0; i < PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT; i++)
    {
      proto[i] = deqnt_affine_to_f32(input_proto[i], zp_proto, scale_proto);
    }
    return validCount;
  }

  int8_t *input = (int8_t *)(*all_input)[input_id];
  int8_t *input_seg = (int8_t *)(*all_input)[input_id + 1];
  int32_t zp = output_zps[input_id];
  float scale = output_scales[input_id];
  int32_t zp_seg = output_zps[input_id + 1];
  float scale_seg = output_scales[input_id + 1];

  int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

  for (int a = 0; a < 3; a++)
  {
    for (int i = 0; i < grid_h; i++)
    {
      for (int j = 0; j < grid_w; j++)
      {
        int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres_i8)
        {
          int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
          int offset_seg = (PROTO_CHANNEL * a) * grid_len + i * grid_w + j;
          int8_t *in_ptr = input + offset;
          int8_t *in_ptr_seg = input_seg + offset_seg;

          float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
          float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
          float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
          float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
          box_x = (box_x + j) * (float)stride;
          box_y = (box_y + i) * (float)stride;
          box_w = box_w * box_w * (float)anchor[a * 2];
          box_h = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          int8_t maxClassProbs = in_ptr[5 * grid_len];
          int maxClassId = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k)
          {
            int8_t prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs)
            {
              maxClassId = k;
              maxClassProbs = prob;
            }
          }

          float box_conf_f32 = deqnt_affine_to_f32(box_confidence, zp, scale);
          float class_prob_f32 = deqnt_affine_to_f32(maxClassProbs, zp, scale);
          float limit_score = box_conf_f32 * class_prob_f32;
          // if (maxClassProbs > thres_i8)
          if (limit_score > threshold)
          {
            for (int k = 0; k < PROTO_CHANNEL; k++)
            {
              float seg_element_fp = deqnt_affine_to_f32(in_ptr_seg[(k)*grid_len], zp_seg, scale_seg);
              segments.push_back(seg_element_fp);
            }

            objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
  return validCount;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
  for (int i = 0; i < validCount; ++i)
  {
    if (order[i] == -1 || classIds[i] != filterId)
    {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j)
    {
      int m = order[j];
      if (m == -1 || classIds[i] != filterId)
      {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      if (iou > threshold)
      {
        order[j] = -1;
      }
    }
  }
  return 0;
}
int clamp(float val, int min, int max)
{
  return val > min ? (val < max ? val : max) : min;
}
int box_reverse(int position, int boundary, int pad, float scale)
{
  return (int)((clamp(position, 0, boundary) - pad) / scale);
}
void matmul_by_cpu_uint8(std::vector<float> &A, float *B, uint8_t *C, int ROWS_A, int COLS_A, int COLS_B)
{

  float temp = 0;
  for (int i = 0; i < ROWS_A; i++)
  {
    for (int j = 0; j < COLS_B; j++)
    {
      temp = 0;
      for (int k = 0; k < COLS_A; k++)
      {
        temp += A[i * COLS_A + k] * B[k * COLS_B + j];
      }
      if (temp > 0)
      {
        C[i * COLS_B + j] = 4;
      }
      else
      {
        C[i * COLS_B + j] = 0;
      }
    }
  }
}
void resize_by_opencv_uint8(uint8_t *input_image, int input_width, int input_height, int boxes_num, uint8_t *output_image, int target_width, int target_height)
{
  for (int b = 0; b < boxes_num; b++)
  {
    cv::Mat src_image(input_height, input_width, CV_8U, &input_image[b * input_width * input_height]);
    cv::Mat dst_image;
    cv::resize(src_image, dst_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
    memcpy(&output_image[b * target_width * target_height], dst_image.data, target_width * target_height * sizeof(uint8_t));
  }
}
void crop_mask_uint8(uint8_t *seg_mask, uint8_t *all_mask_in_one, float *boxes, int boxes_num, int *cls_id, int height, int width)
{
  for (int b = 0; b < boxes_num; b++)
  {
    float x1 = boxes[b * 4 + 0];
    float y1 = boxes[b * 4 + 1];
    float x2 = boxes[b * 4 + 2];
    float y2 = boxes[b * 4 + 3];

    for (int i = 0; i < height; i++)
    {
      for (int j = 0; j < width; j++)
      {
        if (j >= x1 && j < x2 && i >= y1 && i < y2)
        {
          if (all_mask_in_one[i * width + j] == 0)
          {
            if (seg_mask[b * width * height + i * width + j] > 0)
            {
              all_mask_in_one[i * width + j] = (cls_id[b] + 1);
            }
            else
            {
              all_mask_in_one[i * width + j] = 0;
            }
          }
        }
      }
    }
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

  // 计算letterbox参数
  float scale = std::min((float)engine.model_input_attrs_[0].dims[2] / img.cols, (float)engine.model_input_attrs_[0].dims[1] / img.rows);
  int new_width = (int)(img.cols * scale);
  int new_height = (int)(img.rows * scale);
  letterbox_t letter_box;
  letter_box.scale = scale;
  letter_box.x_pad = (engine.model_input_attrs_[0].dims[2] - new_width) / 2;
  letter_box.y_pad = (engine.model_input_attrs_[0].dims[1] - new_height) / 2;

  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(new_width, new_height));
  cv::copyMakeBorder(resized_img, resized_img, letter_box.y_pad, letter_box.y_pad, letter_box.x_pad, letter_box.x_pad,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

  rknn_input inputs[engine.model_io_num_.n_input];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = engine.model_input_attrs_[0].dims[1] * engine.model_input_attrs_[0].dims[2] * engine.model_input_attrs_[0].dims[3];
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;
  inputs[0].buf = resized_img.data;

  // 比较图像数据与二进制文件内容
  std::string binFilePath = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_seg/input_image.bin";
  bool isMatch = compareImageData(img, binFilePath);
  std::cout << "图像数据与二进制文件匹配结果: " << (isMatch ? "一致" : "不一致") << std::endl;

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

  // 比较模型输出与NPZ文件
  std::string npzFilePath = "/home/orangepi/HectorHuang/deploy_percept/examples/data/yolov5_seg/yolov5_seg_output.npz";
  bool outputsMatch = compareOutputsToNpz(outputs, engine.model_io_num_.n_output, npzFilePath);
  std::cout << "模型输出与NPZ文件匹配结果: " << (outputsMatch ? "一致" : "不一致") << std::endl;

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

  object_detect_result_list loaded_results;
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

  int model_in_width = engine.model_input_attrs_[0].dims[2]; // width comes second in NHWC
  int model_in_height = engine.model_input_attrs_[0].dims[1];

  // 使用YoloV5SegPostProcess类进行后处理
  deploy_percept::post_process::YoloV5SegPostProcess::Params params_post;
  params_post.conf_threshold = BOX_THRESH;
  params_post.nms_threshold = NMS_THRESH;
  params_post.obj_class_num = 80;
  params_post.obj_numb_max_size = 128;

  deploy_percept::post_process::YoloV5SegPostProcess seg_processor(params_post);

  // 创建BoxRect用于pads参数
  deploy_percept::post_process::BoxRect pads;
  pads.left = letter_box.x_pad;
  pads.top = letter_box.y_pad;
  pads.right = letter_box.x_pad;
  pads.bottom = letter_box.y_pad;

  // 调用新的后处理类
  bool success = seg_processor.run(
      model_in_width,
      model_in_height,
      output_dims,
      output_scales,
      output_zps,
      &output_buffers,
      pads,
      letter_box.scale,
      orig_img.rows,
      orig_img.cols);

  if (!success)
  {
    printf("Post-processing failed: %s\n", seg_processor.getResult().message.c_str());
    return -1;
  }

  // 获取后处理结果
  auto seg_results = seg_processor.getResult().group;

  // 将 YoloV5SegPostProcess 的结果转换为 object_detect_result_list 格式以便比较和绘制
  object_detect_result_list od_results;
  memset(&od_results, 0x00, sizeof(od_results));
  od_results.count = seg_results.count > OBJ_NUMB_MAX_SIZE ? OBJ_NUMB_MAX_SIZE : seg_results.count;

  for (int i = 0; i < od_results.count; ++i) {
    od_results.results[i].box.left = seg_results.results[i].box.left;
    od_results.results[i].box.top = seg_results.results[i].box.top;
    od_results.results[i].box.right = seg_results.results[i].box.right;
    od_results.results[i].box.bottom = seg_results.results[i].box.bottom;
    od_results.results[i].prop = seg_results.results[i].prop;
    
    // 解析类别ID
    std::string name_str(seg_results.results[i].name);
    size_t pos = name_str.find_last_of('_');
    if (pos != std::string::npos) {
        od_results.results[i].cls_id = std::stoi(name_str.substr(pos + 1));
    } else {
        od_results.results[i].cls_id = 0;  // 默认为0
    }

    // 复制掩码数据
    if (i < seg_results.results_seg.size() && seg_results.results_seg[0].seg_mask != nullptr) {
      // 分配内存并复制掩码数据
      int mask_size = orig_img.rows * orig_img.cols * sizeof(uint8_t);
      od_results.results_seg[i].seg_mask = (uint8_t*)malloc(mask_size);
      if (od_results.results_seg[i].seg_mask != nullptr) {
        memcpy(od_results.results_seg[i].seg_mask, seg_results.results_seg[0].seg_mask, mask_size);
      }
    } else {
      od_results.results_seg[i].seg_mask = nullptr;
    }
  }

  // 比较加载的结果和计算的结果
  printf("Comparing loaded results with computed results...\n");
  bool results_match = compareDetectionResults(loaded_results, od_results, orig_img.cols, orig_img.rows);
  printf("Results match: %s\n", results_match ? "YES" : "NO");

  // 绘制计算得到的检测结果
  cv::Mat result_img = orig_img.clone();
  drawDetectionResults(result_img, od_results);

  std::string computed_out_path = "/home/orangepi/HectorHuang/deploy_percept/tmp/computed_out.jpg";
  printf("Save computed detect result to %s\n", computed_out_path.c_str());
  cv::imwrite(computed_out_path, result_img);

  cv::Mat result_img_2 = orig_img.clone();
  drawDetectionResults(result_img_2, loaded_results);
  std::string computed_out_path_2 = "/home/orangepi/HectorHuang/deploy_percept/tmp/out.jpg";
  printf("Save loaded detect result to %s\n", computed_out_path_2.c_str());
  cv::imwrite(computed_out_path_2, result_img_2);

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

  rknn_outputs_release(engine.ctx_, engine.model_io_num_.n_output, outputs);

  // 释放分配的掩码内存
  for (int i = 0; i < od_results.count; ++i)
  {
    if (od_results.results_seg[i].seg_mask != nullptr)
    {
      free(od_results.results_seg[i].seg_mask);
      od_results.results_seg[i].seg_mask = nullptr;
    }
  }

  return 0;
}