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
// #include "postprocess.h"
#include <sys/time.h>
#include <fstream>
#include <chrono>
#include <yaml-cpp/yaml.h>

#define PERF_WITH_POST 1
#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

#define LABEL_NALE_TXT_PATH "/home/orangepi/HectorHuang/deploy_percept/apps/demo/coco_80_labels_list.txt"

const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};

static char *labels[OBJ_CLASS_NUM];

typedef struct _BOX_RECT
{
  int left;
  int right;
  int top;
  int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
  char name[OBJ_NAME_MAX_SIZE];
  BOX_RECT box;
  float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
  int id;
  int count;
  detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
void deinitPostProcess()
{
  for (int i = 0; i < OBJ_CLASS_NUM; i++)
  {
    if (labels[i] != nullptr)
    {
      free(labels[i]);
      labels[i] = nullptr;
    }
  }
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
inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
  for (int i = 0; i < validCount; ++i)
  {
    int n = order[i];
    if (n == -1 || classIds[n] != filterId)
    {
      continue;
    }
    for (int j = i + 1; j < validCount; ++j)
    {
      int m = order[j];
      if (m == -1 || classIds[m] != filterId)
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
// int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size)
// {
//   im_rect src_rect;
//   im_rect dst_rect;
//   memset(&src_rect, 0, sizeof(src_rect));
//   memset(&dst_rect, 0, sizeof(dst_rect));
//   size_t img_width = image.cols;
//   size_t img_height = image.rows;
//   if (image.type() != CV_8UC3)
//   {
//     printf("source image type is %d!\n", image.type());
//     return -1;
//   }
//   size_t target_width = target_size.width;
//   size_t target_height = target_size.height;
//   src = wrapbuffer_virtualaddr((void *)image.data, img_width, img_height, RK_FORMAT_RGB_888);
//   dst = wrapbuffer_virtualaddr((void *)resized_image.data, target_width, target_height, RK_FORMAT_RGB_888);
//   int ret = imcheck(src, dst, src_rect, dst_rect);
//   if (IM_STATUS_NOERROR != ret)
//   {
//     fprintf(stderr, "rga check error! %s", imStrError((IM_STATUS)ret));
//     return -1;
//   }
//   IM_STATUS STATUS = imresize(src, dst);
//   return 0;
// }
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
  unsigned char *data;
  int ret;

  data = NULL;

  if (NULL == fp)
  {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0)
  {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char *)malloc(sz);
  if (data == NULL)
  {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}
static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp;
  unsigned char *data;

  fp = fopen(filename, "rb");
  if (NULL == fp)
  {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);
  if (size <= 0)
  {
    printf("Get file size failed.\n");
    fclose(fp);
    return NULL;
  }

  rewind(fp); // 重置文件指针到开始位置
  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}
void letterbox(const cv::Mat &image, cv::Mat &padded_image, BOX_RECT &pads, const float scale, const cv::Size &target_size, const cv::Scalar &pad_color)
{
  // 调整图像大小
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(), scale, scale);

  // 计算填充大小
  int pad_width = target_size.width - resized_image.cols;
  int pad_height = target_size.height - resized_image.rows;

  pads.left = pad_width / 2;
  pads.right = pad_width - pads.left;
  pads.top = pad_height / 2;
  pads.bottom = pad_height - pads.top;

  // 在图像周围添加填充
  cv::copyMakeBorder(resized_image, padded_image, pads.top, pads.bottom, pads.left, pads.right, cv::BORDER_CONSTANT, pad_color);
}
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i)
  {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}
char *readLine(FILE *fp, char *buffer, int *len)
{
  int ch;
  int i = 0;
  size_t buff_len = 0;

  buffer = (char *)malloc(buff_len + 1);
  if (!buffer)
    return NULL; // Out of memory

  while ((ch = fgetc(fp)) != '\n' && ch != EOF)
  {
    buff_len++;
    void *tmp = realloc(buffer, buff_len + 1);
    if (tmp == NULL)
    {
      free(buffer);
      return NULL; // Out of memory
    }
    buffer = (char *)tmp;

    buffer[i] = (char)ch;
    i++;
  }
  buffer[i] = '\0';

  *len = buff_len;

  // Detect end
  if (ch == EOF && (i == 0 || ferror(fp)))
  {
    free(buffer);
    return NULL;
  }
  return buffer;
}
int readLines(const char *fileName, char *lines[], int max_line)
{
  FILE *file = fopen(fileName, "r");
  char *s;
  int i = 0;
  int n = 0;

  if (file == NULL)
  {
    printf("Open %s fail!\n", fileName);
    return -1;
  }

  while ((s = readLine(file, s, &n)) != NULL)
  {
    lines[i++] = s;
    if (i >= max_line)
      break;
  }
  fclose(file);
  return i;
}
int loadLabelName(const char *locationFilename, char *label[])
{
  printf("loadLabelName %s\n", locationFilename);
  readLines(locationFilename, label, OBJ_CLASS_NUM);
  return 0;
}
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
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }
static int process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                   int32_t zp, float scale)
{
  int validCount = 0;
  int grid_len = grid_h * grid_w;
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
          int8_t *in_ptr = input + offset;
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
          if (maxClassProbs > thres_i8)
          {
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
int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, BOX_RECT pads, float scale_w, float scale_h, std::vector<int32_t> &qnt_zps,
                 std::vector<float> &qnt_scales, detect_result_group_t *group)
{
  static int init = -1;
  if (init == -1)
  {
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0)
    {
      return -1;
    }

    init = 0;
  }
  memset(group, 0, sizeof(detect_result_group_t));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;

  // stride 8
  int stride0 = 8;
  int grid_h0 = model_in_h / stride0;
  int grid_w0 = model_in_w / stride0;
  int validCount0 = 0;
  validCount0 = process(input0, (int *)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

  // stride 16
  int stride1 = 16;
  int grid_h1 = model_in_h / stride1;
  int grid_w1 = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = process(input1, (int *)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

  // stride 32
  int stride2 = 32;
  int grid_h2 = model_in_h / stride2;
  int grid_w2 = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = process(input2, (int *)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

  int validCount = validCount0 + validCount1 + validCount2;
  // no object detect
  if (validCount <= 0)
  {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i)
  {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set)
  {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  group->count = 0;
  /* box valid detect target */
  for (int i = 0; i < validCount; ++i)
  {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
    {
      continue;
    }
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - pads.left;
    float y1 = filterBoxes[n * 4 + 1] - pads.top;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];

    group->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
    group->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
    group->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
    group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
    group->results[last_count].prop = obj_conf;
    char *label = labels[id];
    strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
    // group->results[last_count].box.top,
    //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;

  return 0;
}

// 保存参数到YAML文件
void saveParamsToYaml(const std::string& filename,
                     int model_h, int model_w,
                     float box_conf_threshold, float nms_threshold,
                     BOX_RECT pads,
                     float scale_w, float scale_h,
                     std::vector<int32_t> qnt_zps,
                     std::vector<float> qnt_scales) {
    YAML::Node params;
    
    params["model_h"] = model_h;
    params["model_w"] = model_w;
    params["box_conf_threshold"] = box_conf_threshold;
    params["nms_threshold"] = nms_threshold;
    
    YAML::Node pads_node;
    pads_node["left"] = pads.left;
    pads_node["top"] = pads.top;
    pads_node["right"] = pads.right;
    pads_node["bottom"] = pads.bottom;
    params["pads"] = pads_node;
    
    params["scale_w"] = scale_w;
    params["scale_h"] = scale_h;
    params["qnt_zps"] = qnt_zps;
    params["qnt_scales"] = qnt_scales;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }
    
    file << params;
    file.close();
    std::cout << "Saved parameters to YAML file: " << filename << std::endl;
}

// 保存检测结果到YAML文件
void saveDetectionResultsToYaml(const std::string& filename, detect_result_group_t* group) {
    YAML::Node results;
    
    results["detection_count"] = group->count;
    
    YAML::Node detections;
    for (int i = 0; i < group->count; i++) {
        detect_result_t* det_result = &(group->results[i]);
        
        YAML::Node detection;
        detection["id"] = i;
        detection["name"] = det_result->name;
        
        YAML::Node box;
        box["left"] = det_result->box.left;
        box["top"] = det_result->box.top;
        box["right"] = det_result->box.right;
        box["bottom"] = det_result->box.bottom;
        detection["box"] = box;
        
        detection["prop"] = det_result->prop;
        
        detections.push_back(detection);
    }
    results["detections"] = detections;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }
    
    file << results;
    file.close();
    std::cout << "Saved detection results to YAML file: " << filename << std::endl;
}

int main()
{
  const char *model_name = "/home/orangepi/HectorHuang/deploy_percept/runs/models/RK3588/yolov5s-640-640.rknn";
  int model_data_size = 0;
  unsigned char *model_data = load_model(model_name, &model_data_size);

  rknn_context ctx;
  auto ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);

  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0)
  {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++)
  {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0)
    {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++)
  {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
  }

  int channel = 3;
  int width = 0;
  int height = 0;
  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
  {
    printf("model is NCHW input fmt\n");
    channel = input_attrs[0].dims[1];
    height = input_attrs[0].dims[2];
    width = input_attrs[0].dims[3];
  }
  else
  {
    printf("model is NHWC input fmt\n");
    height = input_attrs[0].dims[1];
    width = input_attrs[0].dims[2];
    channel = input_attrs[0].dims[3];
  }
  printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = width * height * channel;
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
  BOX_RECT pads;
  memset(&pads, 0, sizeof(BOX_RECT));
  cv::Size target_size(width, height);
  cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
  // 计算缩放比例
  float scale_w = (float)target_size.width / img.cols;
  float scale_h = (float)target_size.height / img.rows;
  std::string option = "letterbox";
  if (img_width != width || img_height != height)
  {
    // 直接缩放采用RGA加速
    if (option == "resize")
    {
      // printf("resize image by rga\n");
      // ret = resize_rga(src, dst, img, resized_img, target_size);
      // if (ret != 0)
      // {
      //   fprintf(stderr, "resize with rga error\n");
      //   return -1;
      // }
      // // 保存预处理图片
      // cv::imwrite("resize_input.jpg", resized_img);
    }
    else if (option == "letterbox")
    {
      printf("resize image with letterbox\n");
      float min_scale = std::min(scale_w, scale_h);
      scale_w = min_scale;
      scale_h = min_scale;
      letterbox(img, resized_img, pads, min_scale, target_size, cv::Scalar(128, 128, 128));
      // 保存预处理图片
      cv::imwrite("/home/orangepi/HectorHuang/deploy_percept/tmp/letterbox_input.jpg", resized_img);
    }
    else
    {
      fprintf(stderr, "Invalid resize option. Use 'resize' or 'letterbox'.\n");
      return -1;
    }
    inputs[0].buf = resized_img.data;
  }
  else
  {
    inputs[0].buf = img.data;
  }
  struct timeval start_time, stop_time;
  gettimeofday(&start_time, NULL);
  rknn_inputs_set(ctx, io_num.n_input, inputs);

  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < io_num.n_output; i++)
  {
    outputs[i].index = i;
    outputs[i].want_float = 0;
  }

  // 执行推理
  ret = rknn_run(ctx, NULL);
  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  gettimeofday(&stop_time, NULL);
  printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

  // 后处理
  detect_result_group_t detect_result_group;
  std::vector<float> out_scales;
  std::vector<int32_t> out_zps;
  for (int i = 0; i < io_num.n_output; ++i)
  {
    out_scales.push_back(output_attrs[i].scale);
    out_zps.push_back(output_attrs[i].zp);
  }
  const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
  const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
  
  // 保存输入数据到NPY文件
  // 计算YOLO输出层的形状 [height/stride, width/stride, channels]
  int8_t* output0_ptr = (int8_t *)outputs[0].buf;
  int8_t* output1_ptr = (int8_t *)outputs[1].buf;
  int8_t* output2_ptr = (int8_t *)outputs[2].buf;
  
  // 假设YOLOv5的输出格式，对于输入尺寸height x width
  int8_t stride0 = 8, stride1 = 16, stride2 = 32;
  int32_t output0_shape[] = {height / stride0, width / stride0, PROP_BOX_SIZE * 3}; // 例如 80x80x255
  int32_t output1_shape[] = {height / stride1, width / stride1, PROP_BOX_SIZE * 3}; // 例如 40x40x255
  int32_t output2_shape[] = {height / stride2, width / stride2, PROP_BOX_SIZE * 3}; // 例如 20x20x255
  
  // 生成时间戳作为文件名的一部分，确保唯一性
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  
  // 使用NPZ格式保存三个输出到一个文件
  std::string npz_filename = "/home/orangepi/HectorHuang/deploy_percept/tmp/yolov5_outputs.npz";
  cnpy::npz_save(npz_filename, "output0", output0_ptr, {output0_shape, output0_shape+3}, "w");  // write mode
  cnpy::npz_save(npz_filename, "output1", output1_ptr, {output1_shape, output1_shape+3}, "a");  // append mode
  cnpy::npz_save(npz_filename, "output2", output2_ptr, {output2_shape, output2_shape+3}, "a");  // append mode
  
  std::cout << "Saved YOLOv5 outputs to NPZ file: " << npz_filename << std::endl;
  
  // 保存参数到YAML文件
  saveParamsToYaml("/home/orangepi/HectorHuang/deploy_percept/tmp/yolov5_params.yaml",
                   height, width, box_conf_threshold, nms_threshold,
                   pads, scale_w, scale_h, out_zps, out_scales);
  
  post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
               box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

  // 保存检测结果到YAML文件
  saveDetectionResultsToYaml("/home/orangepi/HectorHuang/deploy_percept/tmp/yolov5_detect_results.yaml", &detect_result_group);

  // 画框和概率
  char text[256];
  for (int i = 0; i < detect_result_group.count; i++)
  {
    detect_result_t *det_result = &(detect_result_group.results[i]);
    sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
    printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
           det_result->box.right, det_result->box.bottom, det_result->prop);
    int x1 = det_result->box.left;
    int y1 = det_result->box.top;
    int x2 = det_result->box.right;
    int y2 = det_result->box.bottom;
    rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(256, 0, 0, 256), 3);
    putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
  }
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
#if PERF_WITH_POST
    post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                 box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
#endif
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
  }
  gettimeofday(&stop_time, NULL);
  printf("loop count = %d , average run  %f ms\n", test_count,
         (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

  deinitPostProcess();

  // release
  ret = rknn_destroy(ctx);

  if (model_data)
  {
    free(model_data);
  }

  return 0;
}