#ifndef DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP
#define DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP

#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"

namespace deploy_percept {
namespace post_process {

class YoloV5DetectPostProcess : public YoloBasePostProcess {
public:
    // 静态常量定义
    static constexpr int OBJ_NAME_MAX_SIZE = 16;
    static constexpr int OBJ_NUMB_MAX_SIZE = 64;
    static constexpr int OBJ_CLASS_NUM = 80;
    static constexpr float NMS_THRESH = 0.45f;
    static constexpr float BOX_THRESH = 0.25f;
    static constexpr int PROP_BOX_SIZE = (5 + OBJ_CLASS_NUM);

    YoloV5DetectPostProcess(float conf_threshold = BOX_THRESH, float nms_threshold = NMS_THRESH);
    virtual ~YoloV5DetectPostProcess() = default;

    // YOLOv5特定的处理接口
    int process(
        int8_t* input0,
        int8_t* input1, 
        int8_t* input2,
        int model_in_h,
        int model_in_w,
        BoxRect pads,
        float scale_w,
        float scale_h,
        std::vector<int32_t>& qnt_zps,
        std::vector<float>& qnt_scales,
        DetectResultGroup* group
    );

    int processYoloOutput(int8_t* input, int* anchor, int grid_h, int grid_w, 
                         int height, int width, int stride,
                         std::vector<float>& boxes, std::vector<float>& objProbs, 
                         std::vector<int>& classId, float threshold,
                         int32_t zp, float scale);

private:
    // YoloV5特有的一些处理函数
    void quickSortIndices(std::vector<float>& input, int left, int right, std::vector<int>& indices);
    
    // 添加私有成员变量存储阈值
    float conf_threshold_;
    float nms_threshold_;
    
    // 锚框相关
    const int* anchor0;
    const int* anchor1;
    const int* anchor2;
};

} // namespace post_process
} // namespace deploy_percept

#endif // DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP