#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"

#include <cstddef>

namespace deploy_percept
{
    namespace post_process
    {

        ClassificationPostProcessAwnn::ClassificationPostProcessAwnn(const Params &params)
            : params_(params)
        {
            resetResult();
        }

        bool ClassificationPostProcessAwnn::run(const std::vector<TensorView> &inputs)
        {
            resetResult();

            if (inputs.empty() || inputs[0].data == nullptr)
            {
                return false;
            }

            const TensorView &output = inputs[0];
            if (output.dtype != TensorDtype::FP32)
            {
                return false;
            }

            const std::size_t expected_bytes =
                static_cast<std::size_t>(params_.num_classes) * sizeof(float);
            if (output.byte_size < expected_bytes)
            {
                return false;
            }

            const auto *logits = static_cast<const float *>(output.data);
            const int num_classes = params_.num_classes;

            int best_id = 0;
            float best_score = logits[0];
            for (int i = 1; i < num_classes; ++i)
            {
                if (logits[i] > best_score)
                {
                    best_score = logits[i];
                    best_id = i;
                }
            }

            result_.class_id = best_id;
            result_.score = best_score;
            result_.success = true;
            return true;
        }

        void ClassificationPostProcessAwnn::resetResult()
        {
            result_ = Result{};
        }

    } // namespace post_process
} // namespace deploy_percept
