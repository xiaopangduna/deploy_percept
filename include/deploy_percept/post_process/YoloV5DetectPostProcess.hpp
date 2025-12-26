#ifndef DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP
#define DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP

#include <vector>
#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"

namespace deploy_percept {
namespace post_process {

class YoloV5DetectPostProcess : public YoloBasePostProcess {
public:
    // 静态常量定义
    static constexpr int OBJ_NAME_MAX_SIZE = 16;
    static constexpr int OBJ_NUMB_MAX_SIZE = 64;
    static constexpr float NMS_THRESH = 0.45f;
    static constexpr float BOX_THRESH = 0.25f;

    // YOLOv5参数配置结构体
    struct YoloV5Params {
        float conf_threshold = BOX_THRESH;     // 检测置信度阈值
        float nms_threshold = NMS_THRESH;      // NMS阈值
        int obj_class_num = 80;                // 类别数量
        int obj_name_max_size = OBJ_NAME_MAX_SIZE; // 类别名称最大长度
        int obj_numb_max_size = OBJ_NUMB_MAX_SIZE; // 检测框最大数量
        std::vector<std::vector<int>> anchors; // 锚框配置，格式为[[s8], [s16], [s32]]
        
        // 默认锚框值
        YoloV5Params() {
            anchors = {
                {10, 13, 16, 30, 33, 23},           // stride 8
                {30, 61, 62, 45, 59, 119},          // stride 16
                {116, 90, 156, 198, 373, 326}       // stride 32
            };
        }
    };

    // 使用参数结构体的构造函数
    explicit YoloV5DetectPostProcess(const YoloV5Params& params = YoloV5Params{});
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

    // 获取当前参数的方法
    const YoloV5Params& getParams() const { return params_; }
    
    // 更新参数的方法
    void setParams(const YoloV5Params& params) { params_ = params; }

private:
    // YoloV5特有的一些处理函数
    void quickSortIndices(std::vector<float>& input, int left, int right, std::vector<int>& indices);
    
    // 参数配置
    YoloV5Params params_;
    
    // 锚框相关（从参数中获取）
    std::vector<int> anchor0_;
    std::vector<int> anchor1_;
    std::vector<int> anchor2_;

    // 根据当前类别数计算的属性
    static constexpr int PROP_BOX_SIZE = (5 + 80);  // 5 = x, y, w, h, obj_confidence; + 类别数
};

} // namespace post_process
} // namespace deploy_percept

#endif // DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP