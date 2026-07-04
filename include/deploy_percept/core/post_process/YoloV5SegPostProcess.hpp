#pragma once

#include "deploy_percept/core/post_process/YoloDetectPostProcess.hpp"

#include <cstdint>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {
        /** YOLOv5 实例分割后处理（RKNN int8，anchor 三头 + proto） */
        class YoloV5SegPostProcess : public YoloDetectPostProcess
        {
        public:
            struct Params
            {
                float conf_threshold = 0.25f;
                float nms_threshold = 0.45f;
                int obj_class_num = 80;
                int obj_numb_max_size = 64;

                int proto_channel = 32;
                int proto_height = 160;
                int proto_width = 160;

                std::vector<int> anchor_stride8 = YoloV5CocoAnchors::kStride8;
                std::vector<int> anchor_stride16 = YoloV5CocoAnchors::kStride16;
                std::vector<int> anchor_stride32 = YoloV5CocoAnchors::kStride32;

                int prop_box_size = 5;
            };

            using Result = SegmentationResult;

            explicit YoloV5SegPostProcess(const Params &params);
            ~YoloV5SegPostProcess() = default;

            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(
                const std::vector<int8_t *> &outputs,
                int input_image_width,
                int input_image_height,
                const std::vector<std::vector<int>> &output_dims,
                const std::vector<float> &output_scales,
                const std::vector<int32_t> &output_zps);

        private:
            Params params_;
            Result result_{};
            std::vector<uint8_t> matmul_out_;
            std::vector<uint8_t> seg_mask_;
            std::vector<uint8_t> all_mask_in_one_;

            int decodeDetectionHead(
                const std::vector<int8_t *> &all_input,
                int input_id,
                int *anchor,
                int grid_h,
                int grid_w,
                int stride,
                std::vector<float> &boxes,
                std::vector<float> &segments,
                std::vector<float> &objProbs,
                std::vector<int> &classId,
                float threshold,
                const std::vector<float> &output_scales,
                const std::vector<int32_t> &output_zps);
        };
    } // namespace post_process
} // namespace deploy_percept
