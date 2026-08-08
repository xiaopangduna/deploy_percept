#include "deploy_percept/core/post_process/ClassificationPostProcessSigmastar.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        ClassificationPostProcessSigmastar::ClassificationPostProcessSigmastar(const Params &params)
            : params_(params)
        {
            resetResult();
        }

        bool ClassificationPostProcessSigmastar::run(const std::vector<TensorView> &inputs)
        {
            resetResult();

            if (inputs.empty() || inputs[0].data == nullptr || params_.num_classes <= 0)
            {
                return false;
            }

            const TensorView &output = inputs[0];
            const int num_classes = params_.num_classes;
            const float scalar = params_.output_scalar;

            std::vector<float> logits(static_cast<std::size_t>(num_classes));

            if (output.dtype == TensorDtype::INT16)
            {
                const std::size_t expected_bytes =
                    static_cast<std::size_t>(num_classes) * sizeof(std::int16_t);
                if (output.byte_size < expected_bytes)
                {
                    return false;
                }
                const auto *data = static_cast<const std::int16_t *>(output.data);
                for (int i = 0; i < num_classes; ++i)
                {
                    logits[static_cast<std::size_t>(i)] =
                        static_cast<float>(data[i]) * scalar;
                }
            }
            else if (output.dtype == TensorDtype::FP32)
            {
                const std::size_t expected_bytes =
                    static_cast<std::size_t>(num_classes) * sizeof(float);
                if (output.byte_size < expected_bytes)
                {
                    return false;
                }
                const auto *data = static_cast<const float *>(output.data);
                for (int i = 0; i < num_classes; ++i)
                {
                    logits[static_cast<std::size_t>(i)] = data[i] * scalar;
                }
            }
            else
            {
                return false;
            }

            // Softmax（与 OpenDLA ST_Algo_Softmax 一致）
            float sum = 0.f;
            for (int i = 0; i < num_classes; ++i)
            {
                logits[static_cast<std::size_t>(i)] = std::exp(logits[static_cast<std::size_t>(i)]);
                sum += logits[static_cast<std::size_t>(i)];
            }
            if (!(sum > 0.f))
            {
                return false;
            }

            int best_id = 0;
            float best_score = logits[0] / sum;
            for (int i = 1; i < num_classes; ++i)
            {
                const float score = logits[static_cast<std::size_t>(i)] / sum;
                if (score > best_score)
                {
                    best_score = score;
                    best_id = i;
                }
            }

            result_.class_id = best_id;
            result_.score = best_score;
            return true;
        }

        void ClassificationPostProcessSigmastar::resetResult()
        {
            result_ = Result{};
        }

    } // namespace post_process
} // namespace deploy_percept
