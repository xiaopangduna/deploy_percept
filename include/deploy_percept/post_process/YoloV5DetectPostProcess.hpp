#ifndef DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP
#define DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP

#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"

namespace deploy_percept {
namespace post_process {

class YoloV5DetectPostProcess : public YoloBasePostProcess {
public:
    YoloV5DetectPostProcess(float conf_threshold = BOX_THRESH, float nms_threshold = NMS_THRESH);
    virtual ~YoloV5DetectPostProcess() = default;

    // 实现 YoloBasePostProcess 的纯虚函数
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
    ) override;

    int processYoloOutput(int8_t* input, int* anchor, int grid_h, int grid_w, 
                         int height, int width, int stride,
                         std::vector<float>& boxes, std::vector<float>& objProbs, 
                         std::vector<int>& classId, float threshold,
                         int32_t zp, float scale) override;

private:
    // YoloV5特有的一些处理函数
    void quickSortIndices(std::vector<float>& input, int left, int right, std::vector<int>& indices);
    int8_t qntF32ToAffine(float f32, int32_t zp, float scale);
    float deqntAffineToF32(int8_t qnt, int32_t zp, float scale);
    int32_t clip(float val, float min, float max);
};

} // namespace post_process
} // namespace deploy_percept

#endif // DEPLOY_PERCEPT_POST_PROCESS_YOLOV5DETECTPOSTPROCESS_HPP