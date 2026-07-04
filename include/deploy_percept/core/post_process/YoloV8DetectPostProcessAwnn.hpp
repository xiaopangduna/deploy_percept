#pragma once

#include "deploy_percept/core/post_process/YoloDetectPostProcess.hpp"

#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * YOLOv8 检测后处理（AWNN FP32 六路输出，DFL + sigmoid 解码）。
         * inputs[0/1] stride 8 grid/score，[2/3] stride 16，[4/5] stride 32；
         * layout 与 awnpu_model_zoo examples/yolov8 一致。
         */
        class YoloV8DetectPostProcessAwnn : public YoloDetectPostProcess
        {
        public:
            struct Params
            {
                /** 模型输入高宽（VIP 顺序 [W,H,C,N]）；检测框坐标落在此尺寸空间 */
                int model_in_h{0};
                int model_in_w{0};

                float conf_threshold = 0.4f;
                float nms_threshold = 0.45f;
                int obj_class_num = 80;
                int obj_numb_max_size = 128;
            };

            using Result = YoloDetectPostProcess::Result;

            explicit YoloV8DetectPostProcessAwnn(const Params &params);
            ~YoloV8DetectPostProcessAwnn() override = default;

            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(const std::vector<TensorView> &inputs);

        private:
            using Proposal = DetectionProposal;

            bool finalizeDetections(std::vector<Proposal> &proposals);

            void generateProposals(
                int stride,
                const float *feat_grid,
                const float *feat_score,
                float prob_threshold,
                std::vector<Proposal> &objects);

            Params params_;
            Result result_{};
        };

    } // namespace post_process
} // namespace deploy_percept
