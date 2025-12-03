#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    // 图像路径
    std::string imagePath = "/home/xiaopangdun/project/skeleton_cpp/tmp/BDD100k_00001.jpg";
    
    // 读取图像
    cv::Mat image = cv::imread(imagePath);
    
    // 检查图像是否成功加载
    if (image.empty()) {
        std::cerr << "错误：无法读取图像 " << imagePath << std::endl;
        return -1;
    }
    
    std::cout << "成功读取图像: " << imagePath << std::endl;
    std::cout << "图像尺寸: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "通道数: " << image.channels() << std::endl;
    
    // 对图像进行一些处理 - 调整亮度和对比度
    cv::Mat processedImage;
    image.convertTo(processedImage, -1, 1.2, 30); // alpha=1.2(对比度), beta=30(亮度)
    
    // 保存处理后的图像
    std::string outputPath = "/home/xiaopangdun/project/skeleton_cpp/tmp/processed_image.jpg";
    bool saved = cv::imwrite(outputPath, processedImage);
    
    if (saved) {
        std::cout << "处理后的图像已保存到: " << outputPath << std::endl;
    } else {
        std::cerr << "错误：无法保存图像到 " << outputPath << std::endl;
        return -1;
    }

    
    return 0;
}