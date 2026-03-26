#pragma once

#include <cstdint>
#include <vector>
#include <opencv2/opencv.hpp>

#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"

namespace deploy_percept
{
namespace post_process
{

class YoloV8PosePostProcess : public YoloBasePostProcess
{
public:
    struct Params
    {
        float conf_threshold = 0.5f;
        float nms_threshold = 0.4f;
        int obj_class_num = 1;
        int obj_numb_max_size = 128;
        /** 与官方 yolov8_pose 固定网格总数一致（640×640 三尺度） */
        static constexpr int kKeypointAnchorLen = 8400;
    };

    struct Result
    {
        PoseResultGroup group{};
    };

    explicit YoloV8PosePostProcess(const Params &params);
    ~YoloV8PosePostProcess() = default;

    const Params &getParams() const { return params_; }
    const Result &getResult() const { return result_; }

    /**
     * @param outputs 各输出缓冲区指针；前 3 路为 int8 检测头，第 4 路为关键点（见 output_types）
     * @param output_types 与 rknn_tensor_attr.type 一致，用于解析第 4 路（float16 / float32 / int8 等）
     */
    bool run(const std::vector<void *> &outputs,
             int input_image_width,
             int input_image_height,
             std::vector<std::vector<int>> &output_dims,
             std::vector<float> &output_scales,
             std::vector<int32_t> &output_zps,
             const std::vector<int32_t> &output_types);

    void drawPoseResults(cv::Mat &image, const PoseResultGroup &results) const;

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

    static int nmsPose(int valid_count,
                       std::vector<float> &output_locations,
                       std::vector<int> &class_ids,
                       std::vector<int> &order,
                       int filter_id,
                       float threshold);

    static float readKeypointValue(const void *buf,
                                   int kp_index,
                                   int keypoint_j,
                                   int plane,
                                   int32_t tensor_type,
                                   int32_t zp,
                                   float scale);

    void gatherKeypointsForDetection(PoseDetectionObject &out,
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
