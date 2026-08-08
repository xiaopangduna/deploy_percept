#pragma once

#include "deploy_percept/core/post_process/BasePostProcess.hpp"
#include "deploy_percept/common/types.hpp"

#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * 分类后处理（Sigmastar IPU 离线输出）。
         *
         * 对齐 OpenDLA classification demo：INT16 × fScalar → Softmax → argmax Top-1。
         * 亦支持 FP32 logits（此时按 output_scalar 缩放，通常为 1.0）。
         */
        class ClassificationPostProcessSigmastar : public BasePostProcess
        {
        public:
            struct Params
            {
                int num_classes = 0;
                float output_scalar = 1.f;
            };

            using Result = ClassificationResult;

            explicit ClassificationPostProcessSigmastar(const Params &params);
            ~ClassificationPostProcessSigmastar() = default;

            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(const std::vector<TensorView> &inputs);

        private:
            void resetResult();

            Params params_;
            Result result_{};
        };

    } // namespace post_process
} // namespace deploy_percept
