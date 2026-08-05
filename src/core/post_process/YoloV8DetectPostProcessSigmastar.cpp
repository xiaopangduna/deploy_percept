#include "deploy_percept/core/post_process/YoloV8DetectPostProcessSigmastar.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        namespace
        {

            float sigmoid(float x)
            {
                return 1.f / (1.f + std::exp(-x));
            }

            void softmax(const float *x, float *y, int length)
            {
                float sum = 0.f;
                for (int i = 0; i < length; ++i)
                {
                    y[i] = std::exp(x[i]);
                    sum += y[i];
                }
                if (sum <= 0.f)
                {
                    return;
                }
                for (int i = 0; i < length; ++i)
                {
                    y[i] /= sum;
                }
            }

            void pushXyxy(
                float x1,
                float y1,
                float x2,
                float y2,
                float score,
                int label,
                std::vector<YoloDetectPostProcess::DetectionProposal> &proposals)
            {
                YoloDetectPostProcess::DetectionProposal p{};
                p.x = x1;
                p.y = y1;
                p.w = x2 - x1;
                p.h = y2 - y1;
                p.prob = score;
                p.label = label;
                if (p.w > 0.f && p.h > 0.f)
                {
                    proposals.push_back(p);
                }
            }

        } // namespace

        YoloV8DetectPostProcessSigmastar::YoloV8DetectPostProcessSigmastar(const Params &params)
            : params_(params)
        {
        }

        bool YoloV8DetectPostProcessSigmastar::run(const std::vector<TensorView> &inputs)
        {
            if (params_.model_in_h <= 0 || params_.model_in_w <= 0 || inputs.empty())
            {
                return false;
            }
            if (inputs.size() != params_.output_shapes.size() ||
                inputs.size() != params_.output_scalars.size())
            {
                std::fprintf(
                    stderr,
                    "YoloV8DetectPostProcessSigmastar: meta size mismatch "
                    "(views=%zu shapes=%zu scalars=%zu)\n",
                    inputs.size(),
                    params_.output_shapes.size(),
                    params_.output_scalars.size());
                return false;
            }

            resetResult(result_);
            std::vector<Proposal> proposals;

            bool ok = false;
            if (inputs.size() == 1)
            {
                ok = decodeFused(inputs[0], proposals);
            }
            else if (inputs.size() == 3)
            {
                ok = true;
                for (std::size_t i = 0; i < 3; ++i)
                {
                    if (!decodeCpuDflLayer(inputs[i], i, proposals))
                    {
                        ok = false;
                        break;
                    }
                }
            }
            else if (inputs.size() == 5)
            {
                ok = decodePureIpu(inputs, proposals);
            }
            else
            {
                std::fprintf(
                    stderr,
                    "YoloV8DetectPostProcessSigmastar: unsupported output count %zu\n",
                    inputs.size());
                return false;
            }

            if (!ok)
            {
                return false;
            }
            if (proposals.empty())
            {
                return true;
            }
            return finalizeDetections(proposals);
        }

        bool YoloV8DetectPostProcessSigmastar::finalizeDetections(std::vector<Proposal> &proposals)
        {
            std::vector<int> picked;
            applyNmsToProposals(proposals, params_.nms_threshold, picked);

            const int model_w = params_.model_in_w;
            const int model_h = params_.model_in_h;
            result_.detections.clear();

            for (int idx : picked)
            {
                if (result_.detections.size() >= static_cast<std::size_t>(params_.obj_numb_max_size))
                {
                    break;
                }

                Proposal obj = proposals[static_cast<std::size_t>(idx)];
                if (obj.w <= 0.f || obj.h <= 0.f)
                {
                    continue;
                }

                obj.x = std::max(0.f, std::min(obj.x, static_cast<float>(model_w - 1)));
                obj.y = std::max(0.f, std::min(obj.y, static_cast<float>(model_h - 1)));
                const float x2 = std::max(0.f, std::min(obj.x + obj.w, static_cast<float>(model_w)));
                const float y2 = std::max(0.f, std::min(obj.y + obj.h, static_cast<float>(model_h)));
                obj.w = x2 - obj.x;
                obj.h = y2 - obj.y;
                if (obj.w <= 0.f || obj.h <= 0.f)
                {
                    continue;
                }

                Detection det{};
                det.box.left = static_cast<int>(std::floor(obj.x));
                det.box.top = static_cast<int>(std::floor(obj.y));
                det.box.right = static_cast<int>(std::ceil(obj.x + obj.w));
                det.box.bottom = static_cast<int>(std::ceil(obj.y + obj.h));
                det.score = obj.prob;
                det.class_id = obj.label;
                if (det.box.right <= det.box.left || det.box.bottom <= det.box.top)
                {
                    continue;
                }
                result_.detections.push_back(det);
            }
            return true;
        }

        bool YoloV8DetectPostProcessSigmastar::decodeFused(
            const TensorView &view,
            std::vector<Proposal> &proposals)
        {
            if (view.data == nullptr)
            {
                return false;
            }

            const auto &shape = params_.output_shapes[0];
            // layout: [?, C=4+numClasses, Nboxes, ...]
            const std::uint32_t channels = shape[1];
            const std::uint32_t num_boxes = shape[2];
            if (channels < 5 || num_boxes == 0)
            {
                return false;
            }
            const int num_classes = static_cast<int>(channels) - 4;
            const float scalar = params_.output_scalars[0];

            if (view.dtype == TensorDtype::FP32)
            {
                const float *base = static_cast<const float *>(view.data);
                const float *cx_head = base;
                const float *cy_head = base + 1 * num_boxes;
                const float *w_head = base + 2 * num_boxes;
                const float *h_head = base + 3 * num_boxes;

                for (std::uint32_t i = 0; i < num_boxes; ++i)
                {
                    const float *score_head = base + 4 * num_boxes + i;
                    float max_score = -std::numeric_limits<float>::infinity();
                    int max_ind = -1;
                    for (int c = 0; c < num_classes; ++c)
                    {
                        const float s = score_head[c * static_cast<int>(num_boxes)];
                        if (s >= params_.conf_threshold && s > max_score)
                        {
                            max_score = s;
                            max_ind = c;
                        }
                    }
                    if (max_ind < 0)
                    {
                        continue;
                    }
                    const float cx = cx_head[i];
                    const float cy = cy_head[i];
                    const float w = w_head[i];
                    const float h = h_head[i];
                    pushXyxy(cx - w / 2.f, cy - h / 2.f, cx + w / 2.f, cy + h / 2.f, max_score, max_ind, proposals);
                }
                return true;
            }

            if (view.dtype == TensorDtype::INT16)
            {
                const std::int16_t *base = static_cast<const std::int16_t *>(view.data);
                const std::int16_t *cx_head = base;
                const std::int16_t *cy_head = base + 1 * num_boxes;
                const std::int16_t *w_head = base + 2 * num_boxes;
                const std::int16_t *h_head = base + 3 * num_boxes;
                const float thr_q = (scalar != 0.f) ? (params_.conf_threshold / scalar) : params_.conf_threshold;

                for (std::uint32_t i = 0; i < num_boxes; ++i)
                {
                    const std::int16_t *score_head = base + 4 * num_boxes + i;
                    float best_q = (scalar > 0.f) ? -std::numeric_limits<float>::infinity()
                                                  : std::numeric_limits<float>::infinity();
                    int best_ind = -1;

                    for (int c = 0; c < num_classes; ++c)
                    {
                        const float q = static_cast<float>(score_head[c * static_cast<int>(num_boxes)]);
                        if (scalar > 0.f)
                        {
                            if (q >= thr_q && q > best_q)
                            {
                                best_q = q;
                                best_ind = c;
                            }
                        }
                        else
                        {
                            if (q <= thr_q && q < best_q)
                            {
                                best_q = q;
                                best_ind = c;
                            }
                        }
                    }
                    if (best_ind < 0)
                    {
                        continue;
                    }
                    const float cx = cx_head[i] * scalar;
                    const float cy = cy_head[i] * scalar;
                    const float w = w_head[i] * scalar;
                    const float h = h_head[i] * scalar;
                    pushXyxy(
                        cx - w / 2.f,
                        cy - h / 2.f,
                        cx + w / 2.f,
                        cy + h / 2.f,
                        best_q * scalar,
                        best_ind,
                        proposals);
                }
                return true;
            }

            std::fprintf(stderr, "YoloV8DetectPostProcessSigmastar: fused dtype unsupported\n");
            return false;
        }

        bool YoloV8DetectPostProcessSigmastar::decodeCpuDflLayer(
            const TensorView &view,
            std::size_t layer,
            std::vector<Proposal> &proposals)
        {
            if (view.data == nullptr)
            {
                return false;
            }

            const auto &shape = params_.output_shapes[layer];
            const std::uint32_t num_channels = shape[3];
            const std::uint32_t grid_w = shape[2];
            const std::uint32_t grid_h = shape[1];
            if (num_channels == 0 || grid_w == 0 || grid_h == 0)
            {
                return false;
            }

            const int reg_max = params_.reg_max;
            const int num_classes = static_cast<int>(num_channels) - reg_max * 4;
            if (num_classes <= 0)
            {
                return false;
            }

            const std::uint32_t stride_w = static_cast<std::uint32_t>(params_.model_in_w) / grid_w;
            const std::uint32_t stride_h = static_cast<std::uint32_t>(params_.model_in_h) / grid_h;
            if (stride_w == 0 || stride_w != stride_h)
            {
                std::fprintf(
                    stderr,
                    "YoloV8DetectPostProcessSigmastar: DFL stride mismatch "
                    "(w=%u h=%u)\n",
                    stride_w,
                    stride_h);
                return false;
            }

            // OpenDLA layout: per-cell [box DFL (4*regMax)][classes], then next cell
            const std::size_t cell_stride = static_cast<std::size_t>(num_channels);
            const float thr_before_sigmoid =
                std::log(params_.conf_threshold / (1.f - params_.conf_threshold));
            std::vector<float> distribution(static_cast<std::size_t>(reg_max));
            std::vector<float> dist_before(static_cast<std::size_t>(reg_max));

            if (view.dtype == TensorDtype::FP32)
            {
                const float *preds = static_cast<const float *>(view.data);
                for (std::uint32_t i = 0; i < grid_h; ++i)
                {
                    for (std::uint32_t j = 0; j < grid_w; ++j)
                    {
                        float best_raw = -std::numeric_limits<float>::infinity();
                        int best_ind = 0;
                        const float *class_head = preds + 4 * reg_max;
                        for (int k = 0; k < num_classes; ++k)
                        {
                            if (class_head[k] > best_raw)
                            {
                                best_raw = class_head[k];
                                best_ind = k;
                            }
                        }
                        if (best_raw >= thr_before_sigmoid)
                        {
                            const float max_score = sigmoid(best_raw);
                            float ltrb[4] = {};
                            for (int k = 0; k < 4; ++k)
                            {
                                softmax(preds + k * reg_max, distribution.data(), reg_max);
                                float dis = 0.f;
                                for (int d = 0; d < reg_max; ++d)
                                {
                                    dis += static_cast<float>(d) *
                                           distribution[static_cast<std::size_t>(d)];
                                }
                                ltrb[k] = dis * static_cast<float>(stride_w);
                            }
                            const float cx = (static_cast<float>(j) + 0.5f) * static_cast<float>(stride_w);
                            const float cy = (static_cast<float>(i) + 0.5f) * static_cast<float>(stride_h);
                            pushXyxy(
                                cx - ltrb[0],
                                cy - ltrb[1],
                                cx + ltrb[2],
                                cy + ltrb[3],
                                max_score,
                                best_ind,
                                proposals);
                        }
                        preds += cell_stride;
                    }
                }
                return true;
            }

            if (view.dtype == TensorDtype::INT16)
            {
                const std::int16_t *preds = static_cast<const std::int16_t *>(view.data);
                const float scalar = params_.output_scalars[layer];
                const bool prefer_min = scalar < 0.f;

                for (std::uint32_t i = 0; i < grid_h; ++i)
                {
                    for (std::uint32_t j = 0; j < grid_w; ++j)
                    {
                        float best_raw = prefer_min ? std::numeric_limits<float>::infinity()
                                                   : -std::numeric_limits<float>::infinity();
                        int best_ind = 0;
                        for (int k = 0; k < num_classes; ++k)
                        {
                            const float r = static_cast<float>(preds[4 * reg_max + k]);
                            if (prefer_min ? (r < best_raw) : (r > best_raw))
                            {
                                best_raw = r;
                                best_ind = k;
                            }
                        }
                        float max_score = best_raw * scalar;
                        if (max_score >= thr_before_sigmoid)
                        {
                            max_score = sigmoid(max_score);
                            float ltrb[4] = {};
                            for (int k = 0; k < 4; ++k)
                            {
                                for (int d = 0; d < reg_max; ++d)
                                {
                                    dist_before[static_cast<std::size_t>(d)] =
                                        static_cast<float>(preds[k * reg_max + d]) * scalar;
                                }
                                softmax(dist_before.data(), distribution.data(), reg_max);
                                float dis = 0.f;
                                for (int d = 0; d < reg_max; ++d)
                                {
                                    dis += static_cast<float>(d) *
                                           distribution[static_cast<std::size_t>(d)];
                                }
                                ltrb[k] = dis * static_cast<float>(stride_w);
                            }
                            const float cx = (static_cast<float>(j) + 0.5f) * static_cast<float>(stride_w);
                            const float cy = (static_cast<float>(i) + 0.5f) * static_cast<float>(stride_h);
                            pushXyxy(
                                cx - ltrb[0],
                                cy - ltrb[1],
                                cx + ltrb[2],
                                cy + ltrb[3],
                                max_score,
                                best_ind,
                                proposals);
                        }
                        preds += cell_stride;
                    }
                }
                return true;
            }

            std::fprintf(stderr, "YoloV8DetectPostProcessSigmastar: DFL dtype unsupported\n");
            return false;
        }

        bool YoloV8DetectPostProcessSigmastar::decodePureIpu(
            const std::vector<TensorView> &inputs,
            std::vector<Proposal> &proposals)
        {
            if (inputs.size() < 5 || inputs[0].data == nullptr || inputs[2].data == nullptr ||
                inputs[3].data == nullptr || inputs[4].data == nullptr)
            {
                return false;
            }

            const float *bboxes = static_cast<const float *>(inputs[0].data);
            const std::int16_t *class_id =
                inputs[1].data != nullptr ? static_cast<const std::int16_t *>(inputs[1].data)
                                          : nullptr;
            const float *scores = static_cast<const float *>(inputs[2].data);
            const std::int16_t *scores_index = static_cast<const std::int16_t *>(inputs[3].data);
            const std::int16_t *box_num = static_cast<const std::int16_t *>(inputs[4].data);
            const int num_boxes = static_cast<int>(*box_num);
            constexpr int kAllBoxes = 8400;

            for (int i = 0; i < num_boxes; ++i)
            {
                const int box_index = static_cast<int>(scores_index[i]);
                float p_box[4];
                for (int j = 0; j < 4; ++j)
                {
                    p_box[j] = bboxes[kAllBoxes * j + box_index];
                }
                const int r_class = class_id == nullptr ? 0 : static_cast<int>(class_id[box_index]);
                const float r_score = scores[box_index];
                if (r_score < params_.conf_threshold)
                {
                    continue;
                }
                pushXyxy(p_box[0], p_box[1], p_box[2], p_box[3], r_score, r_class, proposals);
            }
            return true;
        }

    } // namespace post_process
} // namespace deploy_percept
