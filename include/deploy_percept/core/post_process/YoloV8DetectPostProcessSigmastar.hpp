#pragma once

#include "deploy_percept/core/post_process/YoloDetectPostProcess.hpp"

#include <cstdint>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * YOLOv8 检测后处理（Sigmastar IPU 离线输出）。
         *
         * 支持 OpenDLA / pcupid 三种输出形态：
         *   - 1 路：IPU 已融合 decode（FP32 / INT16 + scalar）
         *   - 3 路：CPU DFL decode（每层 INT16/FP32）
         *   - 5 路：IPU filter（boxes/classes/scores/score_index/score_number）
         *     pcupid with_filter 为 FP32、boxes=[1,N,4]；旧 OpenDLA 为 INT16 index/num、boxes=[1,4,N]
         *
         * 调用前用 extract_shapes_from_MI_IPU_TensorDesc_t /
         * extract_fscalars_from_MI_IPU_TensorDesc_t 填写。
         */
        class YoloV8DetectPostProcessSigmastar : public YoloDetectPostProcess
        {
        public:
            struct Params
            {
                int model_in_h{0};
                int model_in_w{0};
                float conf_threshold = 0.5f;
                float nms_threshold = 0.5f;
                int obj_numb_max_size = 128;
                int reg_max = 16;

                /** 与 run() 的 inputs 下标对齐（extract_shapes_from_MI_IPU_TensorDesc_t /
                 *  extract_fscalars_from_MI_IPU_TensorDesc_t） */
                std::vector<std::vector<std::uint32_t>> output_shapes;
                std::vector<float> output_scalars;
            };

            using Result = YoloDetectPostProcess::Result;

            explicit YoloV8DetectPostProcessSigmastar(const Params &params);
            ~YoloV8DetectPostProcessSigmastar() = default;

            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(const std::vector<TensorView> &inputs);

        private:
            using Proposal = DetectionProposal;

            bool finalizeDetections(std::vector<Proposal> &proposals);
            bool decodeFused(const TensorView &view, std::vector<Proposal> &proposals);
            bool decodeCpuDflLayer(
                const TensorView &view,
                std::size_t layer,
                std::vector<Proposal> &proposals);
            bool decodePureIpu(const std::vector<TensorView> &inputs, std::vector<Proposal> &proposals);

            Params params_;
            Result result_{};
        };

    } // namespace post_process
} // namespace deploy_percept
