#pragma once

#include "deploy_percept/core/post_process/YoloDetectPostProcess.hpp"

#include <cstdint>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {
        /** YOLOv8 实例分割后处理（RKNN int8，DFL 三头 + proto） */
        class YoloV8SegPostProcess : public YoloDetectPostProcess
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

                int prop_box_size = 5;
            };

            using Result = SegmentationResult;

            explicit YoloV8SegPostProcess(const Params &params);
            ~YoloV8SegPostProcess() = default;

            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(
                const std::vector<int8_t *> &outputs,
                int input_image_width,
                int input_image_height,
                std::vector<std::vector<int>> &output_dims,
                std::vector<float> &output_scales,
                std::vector<int32_t> &output_zps);

        private:
            Params params_;
            Result result_{};
            std::vector<uint8_t> matmul_out_;
            std::vector<uint8_t> seg_mask_;
            std::vector<uint8_t> all_mask_in_one_;

            int decodeDetectionAndSegmentationHead(
                const std::vector<int8_t *> *all_input,
                int input_id,
                int grid_h,
                int grid_w,
                int stride,
                std::vector<float> &boxes,
                std::vector<float> &segments,
                std::vector<float> &objProbs,
                std::vector<int> &classId,
                float threshold,
                int dfl_len,
                std::vector<float> &output_scales,
                std::vector<int32_t> &output_zps);
        };
    } // namespace post_process
} // namespace deploy_percept
