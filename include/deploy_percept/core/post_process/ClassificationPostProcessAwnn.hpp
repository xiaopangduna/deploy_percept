#pragma once

#include "deploy_percept/common/types.hpp"

#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * ImageNet 分类后处理（AWNN FP32 单路 logits 输出）。
         * 对 logits 做 argmax，输出 Top-1 class_id / score。
         */
        class ClassificationPostProcessAwnn
        {
        public:
            struct Params
            {
                int num_classes = 1000;
            };

            struct Result
            {
                bool success = false;
                int class_id = -1;
                float score = 0.f;
            };

            explicit ClassificationPostProcessAwnn(const Params &params);
            ~ClassificationPostProcessAwnn() = default;

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
