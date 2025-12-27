---
trigger: manual
---

### 代码风格
1.***头文件导入**
    - 使用 #pragma once

### 测试规范
1. **框架与工具**：
   - 测试框架：gtest
   - 路径管理：filesystem


### 代码规范
1. **类的构建**：
```
示例：
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace ud_calib
{
    /**
     * @brief 相机标定器类，用于计算相机外参
     *
     * 该类独立于ROS节点，便于进行单元测试
     */
    class CameraCalibrator
    {
    public:
        struct Params
        {
            cv::Mat camera_matrix;
            cv::Mat distortion_coeffs;
            cv::Mat object_points;

            /**
             * @brief 从YAML节点加载标定参数
             * @param node YAML节点
             * @return 加载是否成功
             */
            bool loadFromYAML(const YAML::Node &node);
        };

        struct Results
        {
            cv::Mat rvec;         ///< 旋转向量
            cv::Mat tvec;         ///< 平移向量
            bool success = false; ///< 标定是否成功
        };
        const Results &getResults() const { return results_; };
        const Params &getParams() const { return params_; };

        explicit CameraCalibrator(const Params &params);

        /**
         * @brief 计算相机外参
         * @param sorted_rectangles 排序后的图像点坐标
         * @param undistort_points 是否需要对图像点进行去畸变处理
         * @return 标定结果
         */
        bool computeExtrinsics(const std::vector<std::vector<cv::Point2f>> &sorted_rectangles,
                               const cv::Mat &object_points,
                               const cv::Mat &camera_matrix,
                               const cv::Mat &distortion_coeffs,
                               bool undistort_points);

    private:
        Params params_;
        Results results_;
    };
}
```