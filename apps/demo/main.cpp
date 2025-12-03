#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "rknn_api.h"

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
    if (size <= 0) {
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

int main()
{
    const char *model_name = "runs/models/RK3588/yolov5s-640-640.rknn";
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_name, &model_data_size);

    // std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // // 图像路径
    // std::string imagePath = "/home/xiaopangdun/project/skeleton_cpp/tmp/BDD100k_00001.jpg";

    // // 读取图像
    // cv::Mat image = cv::imread(imagePath);

    // // 检查图像是否成功加载
    // if (image.empty()) {
    //     std::cerr << "错误：无法读取图像 " << imagePath << std::endl;
    //     return -1;
    // }

    // std::cout << "成功读取图像: " << imagePath << std::endl;
    // std::cout << "图像尺寸: " << image.cols << "x" << image.rows << std::endl;
    // std::cout << "通道数: " << image.channels() << std::endl;

    // // 对图像进行一些处理 - 调整亮度和对比度
    // cv::Mat processedImage;
    // image.convertTo(processedImage, -1, 1.2, 30); // alpha=1.2(对比度), beta=30(亮度)

    // // 保存处理后的图像
    // std::string outputPath = "/home/xiaopangdun/project/skeleton_cpp/tmp/processed_image.jpg";
    // bool saved = cv::imwrite(outputPath, processedImage);

    // if (saved) {
    //     std::cout << "处理后的图像已保存到: " << outputPath << std::endl;
    // } else {
    //     std::cerr << "错误：无法保存图像到 " << outputPath << std::endl;
    //     return -1;
    // }

    return 0;
}