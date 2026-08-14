#pragma once

#include "deploy_percept/core/post_process/YoloDetectPostProcess.hpp"
#include "deploy_percept/common/types.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * YOLOv8 实例分割后处理（Sigmastar IPU 离线输出）。
         *
         * 对齐 OpenDLA segmentation/yolov8：
         *   - 2 路：fused det+mask coeffs + proto（FP32 / INT16 + scalar）
         *   - Softmax 前 logit 阈值：log(mask_thr / (1 - mask_thr))
         *
         * label_mask：与 model 输入同分辨率，0=背景，非 0=class_id+1（供 drawSegmentationResults）。
         */
        class YoloV8SegPostProcessSigmastar : public YoloDetectPostProcess
        {
        public:
            struct Params
            {
                int model_in_h{0};
                int model_in_w{0};
                float conf_threshold = 0.5f;
                float nms_threshold = 0.5f;
                float mask_threshold = 0.5f;
                int num_classes = 80;
                int obj_numb_max_size = 128;

                /** 与 run() 的 inputs 下标对齐（ipu_output_shapes / ipu_output_scalars(desc)） */
                std::vector<std::array<std::uint32_t, 6>> output_shapes;
                std::vector<float> output_scalars;
            };

            using Result = SegmentationResult;

            explicit YoloV8SegPostProcessSigmastar(const Params &params);
            ~YoloV8SegPostProcessSigmastar() = default;

            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(const std::vector<TensorView> &inputs);

        private:
            static constexpr int kMaxMaskProtosChannel = 32;

            struct SegProposal
            {
                float x{0.f};
                float y{0.f};
                float w{0.f};
                float h{0.f};
                int label{0};
                float prob{0.f};
                std::array<float, kMaxMaskProtosChannel> mask_weights{};
            };

            bool decodeFusedSeg(const TensorView &view, std::vector<SegProposal> &proposals);
            void applyClassAwareNms(std::vector<SegProposal> &proposals) const;
            bool buildMasks(
                const TensorView &proto_view,
                const std::vector<SegProposal> &kept,
                int mask_c,
                int mask_h,
                int mask_w);

            Params params_;
            Result result_{};
        };

    } // namespace post_process
} // namespace deploy_percept
