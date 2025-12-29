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

// 添加deploy_percept相关头文件
#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"

#define PERF_WITH_POST 1
#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

#define LABEL_NALE_TXT_PATH "/home/orangepi/HectorHuang/deploy_percept/apps/demo/coco_80_labels_list.txt"

// 定义原始代码中使用的类型，与项目类型兼容
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

static char *labels[OBJ_CLASS_NUM];

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

// 添加deinitPostProcess函数定义
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

  // 后处理 - 使用YoloV5DetectPostProcess类
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
  
  // 保存输入数据到NPZ文件
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
  
  // 使用YoloV5DetectPostProcess类进行后处理
  deploy_percept::post_process::YoloV5DetectPostProcess::Params params;
  params.conf_threshold = BOX_THRESH;  // 使用与原始函数相同的阈值
  params.nms_threshold = NMS_THRESH;
  params.obj_class_num = OBJ_CLASS_NUM;
  params.obj_name_max_size = OBJ_NAME_MAX_SIZE;
  params.obj_numb_max_size = OBJ_NUMB_MAX_SIZE;
  
  deploy_percept::post_process::YoloV5DetectPostProcess processor(params);
  
  // 转换BOX_RECT类型到项目类型
  deploy_percept::post_process::BoxRect new_pads;
  new_pads.left = pads.left;
  new_pads.right = pads.right;
  new_pads.top = pads.top;
  new_pads.bottom = pads.bottom;
  
  processor.run((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, 
                height, width, new_pads, scale_w, scale_h, out_zps, out_scales);

  // 保存检测结果到YAML文件
  // 将processor的结果复制到detect_result_group中以保持兼容性
  const auto& result_wrapper = processor.getResult();
  const auto& new_detect_result_group = result_wrapper.group;
  detect_result_group.count = new_detect_result_group.count;
  for (int i = 0; i < new_detect_result_group.count; i++) {
      strncpy(detect_result_group.results[i].name, new_detect_result_group.results[i].name, OBJ_NAME_MAX_SIZE-1);
      detect_result_group.results[i].name[OBJ_NAME_MAX_SIZE-1] = '\0';
      detect_result_group.results[i].box.left = new_detect_result_group.results[i].box.left;
      detect_result_group.results[i].box.top = new_detect_result_group.results[i].box.top;
      detect_result_group.results[i].box.right = new_detect_result_group.results[i].box.right;
      detect_result_group.results[i].box.bottom = new_detect_result_group.results[i].box.bottom;
      detect_result_group.results[i].prop = new_detect_result_group.results[i].prop;
  }
  
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
    // 使用YoloV5DetectPostProcess类进行后处理
    deploy_percept::post_process::YoloV5DetectPostProcess::Params params;
    params.conf_threshold = BOX_THRESH;  // 使用与原始函数相同的阈值
    params.nms_threshold = NMS_THRESH;
    
    deploy_percept::post_process::YoloV5DetectPostProcess processor(params);
    
    // 转换BOX_RECT类型到项目类型
    deploy_percept::post_process::BoxRect new_pads;
    new_pads.left = pads.left;
    new_pads.right = pads.right;
    new_pads.top = pads.top;
    new_pads.bottom = pads.bottom;
    
    processor.run((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, 
                  height, width, new_pads, scale_w, scale_h, out_zps, out_scales);
    
    // 获取结果用于验证
    const auto& result_wrapper = processor.getResult();
    const auto& new_detect_result_group = result_wrapper.group;
    
    // 验证结果一致性
    printf("Processor: %d detections\n", new_detect_result_group.count);
    for (int j = 0; j < new_detect_result_group.count; j++) {
        printf("Result[%d]: %s (%d, %d, %d, %d) conf=%.3f\n", j,
               new_detect_result_group.results[j].name,
               new_detect_result_group.results[j].box.left, new_detect_result_group.results[j].box.top,
               new_detect_result_group.results[j].box.right, new_detect_result_group.results[j].box.bottom,
               new_detect_result_group.results[j].prop);
    }
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