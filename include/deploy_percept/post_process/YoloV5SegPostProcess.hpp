#pragma once

#include <vector>
#include <cstring>
#include <string>
#include <opencv2/opencv.hpp>
#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/post_process/types.hpp"

namespace deploy_percept
{
    namespace post_process
    {
        class YoloV5SegPostProcess : public YoloBasePostProcess
        {
        public:
            // 参数配置结构体
            struct Params
            {
                float conf_threshold = 0.25f;
                float nms_threshold = 0.45f;
                int obj_class_num = 80;
                int obj_name_max_size = 16;
                int obj_numb_max_size = 64;

                // 分割相关参数
                int proto_channel = 32;
                int proto_height = 160;
                int proto_weight = 160;

                // Anchor参数
                std::vector<int> anchor_stride8 = {10, 13, 16, 30, 33, 23};
                std::vector<int> anchor_stride16 = {30, 61, 62, 45, 59, 119};
                std::vector<int> anchor_stride32 = {116, 90, 156, 198, 373, 326};

                // 模型结构参数
                int prop_box_size = 5; // 边界框坐标(xywh) + 置信度
            };

            // 结果结构体
            struct Result
            {
                ResultGroup group{}; // 使用统一的ResultGroup
                bool success = false;
                std::string message{}; // 可选的详细信息
            };

            // 使用参数结构体的构造函数
            explicit YoloV5SegPostProcess(const Params &params);
            ~YoloV5SegPostProcess() = default;
            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(
                const std::vector<int8_t*>& outputs,
                int input_image_width,
                int input_image_height,
                const std::vector<std::vector<int>>& output_dims,
                const std::vector<float>& output_scales,
                const std::vector<int32_t>& output_zps);

        private:
            Params params_;
            Result result_{};
            std::vector<uint8_t> matmul_out_;
            std::vector<uint8_t> seg_mask_;
            std::vector<uint8_t> all_mask_in_one_;

            int decodeDetectionHead(const std::vector<int8_t*>& all_input, int input_id, int *anchor, int grid_h, int grid_w,
                           int stride,
                           std::vector<float> &boxes, std::vector<float> &segments,
                           std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                           const std::vector<std::vector<int>>& output_dims, const std::vector<float>& output_scales,
                           const std::vector<int32_t>& output_zps);

            // 新增：处理NMS后检测结果的函数
            void collectDetectionsAfterNMS(
                const std::vector<int> &indexArray,
                const std::vector<float> &filterBoxes,
                const std::vector<int> &classId,
                const std::vector<float> &objProbs,
                const std::vector<float> &filterSegments,
                int validCount,
                std::vector<float> &filterSegments_by_nms,
                int &last_count);
        };
    } // namespace post_process
} // namespace deploy_percept