#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"

#include "deploy_percept/data/imagenet_labels.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        ClassificationPostProcessAwnn::ClassificationPostProcessAwnn(const Params &params)
            : params_(params)
        {
            resetResult();
        }

        void ClassificationPostProcessAwnn::resetResult()
        {
            result_ = ClassificationResult{};
            result_.success = false;
            result_.message.clear();
            result_.top_k.clear();
        }

        bool ClassificationPostProcessAwnn::run(const std::vector<TensorView> &inputs)
        {
            resetResult();

            if (inputs.empty() || inputs[0].data == nullptr)
            {
                result_.message = "missing classification output tensor";
                return false;
            }

            const TensorView &output = inputs[0];
            if (output.dtype != TensorDtype::FP32)
            {
                result_.message = "classification output must be FP32";
                return false;
            }

            const std::size_t expected_bytes =
                static_cast<std::size_t>(params_.num_classes) * sizeof(float);
            if (output.byte_size < expected_bytes)
            {
                result_.message = "classification output byte_size too small";
                return false;
            }

            const auto *logits = static_cast<const float *>(output.data);
            const int num_classes = params_.num_classes;
            const int top_k = std::min(params_.top_k, num_classes);

            std::vector<int> indices(static_cast<std::size_t>(num_classes));
            for (int i = 0; i < num_classes; ++i)
            {
                indices[static_cast<std::size_t>(i)] = i;
            }

            std::partial_sort(
                indices.begin(),
                indices.begin() + top_k,
                indices.end(),
                [logits](const int lhs, const int rhs)
                {
                    return logits[lhs] > logits[rhs];
                });

            result_.top_k.reserve(static_cast<std::size_t>(top_k));
            for (int rank = 0; rank < top_k; ++rank)
            {
                const int class_id = indices[static_cast<std::size_t>(rank)];
                ClassificationEntry entry;
                entry.class_id = class_id;
                entry.prob = logits[class_id];
                entry.label = data::imagenet_label(class_id);
                result_.top_k.push_back(std::move(entry));
            }

            result_.success = true;
            return true;
        }

    } // namespace post_process
} // namespace deploy_percept
