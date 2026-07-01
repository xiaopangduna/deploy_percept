#pragma once

#include <string>
#include <vector>

#include "deploy_percept/post_process/YoloBasePostProcess.hpp"
#include "deploy_percept/types.hpp"

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * YOLOv8 检测后处理（AWNN FP32 六路输出，DFL + sigmoid 解码）。
         * inputs[0/1] stride 8 grid/score，[2/3] stride 16，[4/5] stride 32；
         * layout 与 awnpu_model_zoo examples/yolov8 一致。
         */
        class YoloV8DetectPostProcessAwnn : public YoloBasePostProcess
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
                int obj_name_max_size = 16;
                int obj_numb_max_size = 128;
            };

            struct Result
            {
                ResultGroup group{};
                bool success = false;
                std::string message{};
            };

            explicit YoloV8DetectPostProcessAwnn(const Params &params);
            ~YoloV8DetectPostProcessAwnn() = default;

            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(const std::vector<TensorView> &inputs);

        private:
            struct Proposal
            {
                float x{0.f};
                float y{0.f};
                float w{0.f};
                float h{0.f};
                int label{0};
                float prob{0.f};
            };

            void resetResult();

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
