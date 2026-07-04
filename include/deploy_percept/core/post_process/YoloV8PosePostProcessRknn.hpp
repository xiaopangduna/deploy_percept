#pragma once

#include "deploy_percept/core/post_process/YoloDetectPostProcess.hpp"

#include <cstdint>
#include <vector>

namespace deploy_percept
{
namespace post_process
{

/** YOLOv8 姿态估计后处理（RKNN，int8 检测头 + 关键点张量） */
class YoloV8PosePostProcessRknn : public YoloDetectPostProcess
{
public:
    struct Params
    {
        float conf_threshold = 0.5f;
        float nms_threshold = 0.4f;
        int obj_class_num = 1;
        int obj_numb_max_size = 128;
        static constexpr int kKeypointAnchorLen = 8400;
        static constexpr int kPoseBoxStride = 5;
    };

    using Result = PoseResult;

    explicit YoloV8PosePostProcessRknn(const Params &params);
    ~YoloV8PosePostProcessRknn() = default;

    const Params &getParams() const { return params_; }
    const Result &getResult() const { return result_; }

    bool run(const std::vector<void *> &outputs,
             int input_image_width,
             int input_image_height,
             std::vector<std::vector<int>> &output_dims,
             std::vector<float> &output_scales,
             std::vector<int32_t> &output_zps,
             const std::vector<int32_t> &output_types);

private:
    Params params_;
    Result result_{};

    static int decodeHeadInt8(const int8_t *input,
                              int grid_h,
                              int grid_w,
                              int stride,
                              int index_base,
                              std::vector<float> &boxes,
                              std::vector<float> &box_scores,
                              std::vector<int> &class_id,
                              float conf_threshold,
                              int32_t zp,
                              float scale,
                              int obj_class_num);

    static float readKeypointValue(const void *buf,
                                   int kp_index,
                                   int keypoint_j,
                                   int plane,
                                   int32_t tensor_type,
                                   int32_t zp,
                                   float scale);

    void gatherKeypointsForDetection(PoseDetection &out,
                                     const void *kpt_buf,
                                     int32_t kpt_type,
                                     int32_t kpt_zp,
                                     float kpt_scale,
                                     int keypoints_index,
                                     int model_in_w,
                                     int model_in_h,
                                     float letter_scale,
                                     float x_pad,
                                     float y_pad);
};

} // namespace post_process
} // namespace deploy_percept
