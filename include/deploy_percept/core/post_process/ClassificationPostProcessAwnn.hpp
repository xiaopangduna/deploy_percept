#pragma once

#include "deploy_percept/common/types.hpp"

#include <string>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        /**
         * ImageNet 分类 Top-K 后处理（AWNN FP32 单路 logits 输出）。
         * 与 awnpu_model_zoo examples/mobilenetv2 class_post.cpp 语义一致。
         */
        class ClassificationPostProcessAwnn
        {
        public:
            struct Params
            {
                int num_classes = 1000;
                int top_k = 5;
            };

            explicit ClassificationPostProcessAwnn(const Params &params);
            ~ClassificationPostProcessAwnn() = default;

            const Params &getParams() const { return params_; }
            const ClassificationResult &getResult() const { return result_; }

            bool run(const std::vector<TensorView> &inputs);

        private:
            void resetResult();

            Params params_;
            ClassificationResult result_{};
        };

    } // namespace post_process
} // namespace deploy_percept
