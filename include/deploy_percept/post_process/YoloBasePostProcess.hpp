#ifndef DEPLOY_PERCEPT_POST_PROCESS_YOLOBASEPOSTPROCESS_HPP
#define DEPLOY_PERCEPT_POST_PROCESS_YOLOBASEPOSTPROCESS_HPP

#include "deploy_percept/post_process/BasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"
#include <vector>
#include <string>

namespace deploy_percept {
namespace post_process {

class YoloBasePostProcess : public BasePostProcess {
public:
    // 静态常量定义
    static constexpr int OBJ_NAME_MAX_SIZE = 16;
    static constexpr int OBJ_NUMB_MAX_SIZE = 64;
    static constexpr int OBJ_CLASS_NUM = 80;
    static constexpr float NMS_THRESH = 0.45f;
    static constexpr float BOX_THRESH = 0.25f;
    static constexpr int PROP_BOX_SIZE = (5 + OBJ_CLASS_NUM);

    YoloBasePostProcess(float conf_threshold = BOX_THRESH, float nms_threshold = NMS_THRESH);
    virtual ~YoloBasePostProcess() = default;

    // YOLO系列通用的处理接口
    virtual int process(
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
    ) = 0;

    // 通用的clamp函数 - 内联实现
    static inline int clamp(float val, int min, int max) {
        return val > min ? (val < max ? static_cast<int>(val) : max) : min;
    }

    // 通用的量化和反量化函数
    static int8_t qntF32ToAffine(float f32, int32_t zp, float scale);
    static float deqntAffineToF32(int8_t qnt, int32_t zp, float scale);
    static int32_t clip(float val, float min, float max);

    // 通用的NMS实现
    int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, 
            std::vector<int>& order, int filterId, float threshold);

    // 通用的边界框重叠计算
    static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, 
                                float xmin1, float ymin1, float xmax1, float ymax1);

protected:
    // Yolo系列通用的处理逻辑
    virtual int processYoloOutput(int8_t* input, int* anchor, int grid_h, int grid_w, 
                                 int height, int width, int stride,
                                 std::vector<float>& boxes, std::vector<float>& objProbs, 
                                 std::vector<int>& classId, float threshold,
                                 int32_t zp, float scale) = 0;

    // 锚框相关
    const int* anchor0;
    const int* anchor1;
    const int* anchor2;
    
    float conf_threshold_;
    float nms_threshold_;
};

} // namespace post_process
} // namespace deploy_percept

#endif // DEPLOY_PERCEPT_POST_PROCESS_YOLOBASEPOSTPROCESS_HPP