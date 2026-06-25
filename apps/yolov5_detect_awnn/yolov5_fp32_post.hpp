#pragma once

#include <opencv2/core.hpp>
#include <vector>

/** YOLOv5 FP32 后处理（对齐 tmp/ai-sdk/examples/yolov5/yolov5_post_process.cpp） */
namespace yolov5_fp32_post
{

struct Detection
{
    float x0{0.f};
    float y0{0.f};
    float x1{0.f};
    float y1{0.f};
    int label{0};
    float score{0.f};
};

/** outputs[0]=stride8, outputs[1]=stride16, outputs[2]=stride32 */
int run(
    float *outputs[3],
    int letterbox_w,
    int letterbox_h,
    std::vector<Detection> &detections,
    float conf_threshold = 0.25f,
    float nms_threshold = 0.45f);

void draw(cv::Mat &bgr, const std::vector<Detection> &detections);

} // namespace yolov5_fp32_post
