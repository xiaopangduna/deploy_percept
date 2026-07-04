#include "deploy_percept/core/post_process/YoloV8DetectPostProcessAwnn.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        namespace
        {

            constexpr int kRegMax = 16;
            constexpr int kGridChannels = kRegMax * 4;

            struct StrideTensors
            {
                const float *grid{nullptr};
                const float *score{nullptr};
            };

            float softmaxExpectation(const float *src, float *dst, int length)
            {
                float alpha = -FLT_MAX;
                for (int c = 0; c < length; ++c)
                {
                    alpha = std::max(alpha, src[c]);
                }

                float denominator = 0.f;
                float dis_sum = 0.f;
                for (int i = 0; i < length; ++i)
                {
                    dst[i] = std::exp(src[i] - alpha);
                    denominator += dst[i];
                }
                for (int i = 0; i < length; ++i)
                {
                    dst[i] /= denominator;
                    dis_sum += static_cast<float>(i) * dst[i];
                }
                return dis_sum;
            }

            bool resolveStrideTensors(
                const std::vector<TensorView> &inputs,
                int model_w,
                int model_h,
                int num_class,
                StrideTensors &s8,
                StrideTensors &s16,
                StrideTensors &s32)
            {
                std::vector<std::pair<std::size_t, const float *>> tensors;
                tensors.reserve(inputs.size());
                for (const TensorView &view : inputs)
                {
                    if (view.data == nullptr || view.dtype != TensorDtype::FP32)
                    {
                        continue;
                    }
                    tensors.emplace_back(view.byte_size / sizeof(float), static_cast<const float *>(view.data));
                }

                if (tensors.size() < 6)
                {
                    return false;
                }

                auto pick_stride = [&](int stride, StrideTensors &out) -> bool {
                    const int grid_w = model_w / stride;
                    const int grid_h = model_h / stride;
                    const std::size_t grid_size = static_cast<std::size_t>(grid_w * grid_h);

                    auto find_by_channels = [&](int channels) -> const float * {
                        for (const auto &item : tensors)
                        {
                            if (grid_size == 0 || item.first % grid_size != 0)
                            {
                                continue;
                            }
                            if (item.first / grid_size == static_cast<std::size_t>(channels))
                            {
                                return item.second;
                            }
                        }
                        return nullptr;
                    };

                    out.grid = find_by_channels(kGridChannels);
                    out.score = find_by_channels(num_class);

                    if (out.grid == nullptr || out.score == nullptr)
                    {
                        return false;
                    }
                    return true;
                };

                if (!pick_stride(8, s8) || !pick_stride(16, s16) || !pick_stride(32, s32))
                {
                    return false;
                }
                return true;
            }

        } // namespace

        YoloV8DetectPostProcessAwnn::YoloV8DetectPostProcessAwnn(const Params &params)
            : params_(params)
        {
        }

        bool YoloV8DetectPostProcessAwnn::run(const std::vector<TensorView> &inputs)
        {
            if (params_.model_in_h <= 0 || params_.model_in_w <= 0)
            {
                result_.success = false;
                return false;
            }

            if (inputs.size() != 6)
            {
                result_.success = false;
                return false;
            }

            for (std::size_t i = 0; i < inputs.size(); ++i)
            {
                const TensorView &view = inputs[i];
                if (view.data == nullptr)
                {
                    result_.success = false;
                    return false;
                }
                if (view.dtype != TensorDtype::FP32)
                {
                    result_.success = false;
                    return false;
                }
            }

            StrideTensors s8{};
            StrideTensors s16{};
            StrideTensors s32{};
            if (!resolveStrideTensors(
                    inputs,
                    params_.model_in_w,
                    params_.model_in_h,
                    params_.obj_class_num,
                    s8,
                    s16,
                    s32))
            {
                result_.success = false;
                return false;
            }

            resetResult(result_);

            std::vector<Proposal> proposals;
            generateProposals(8, s8.grid, s8.score, params_.conf_threshold, proposals);
            generateProposals(16, s16.grid, s16.score, params_.conf_threshold, proposals);
            generateProposals(32, s32.grid, s32.score, params_.conf_threshold, proposals);

            return finalizeDetections(proposals);
        }

        bool YoloV8DetectPostProcessAwnn::finalizeDetections(std::vector<Proposal> &proposals)
        {
            if (proposals.empty())
            {
                return false;
            }

            std::vector<int> picked;
            applyNmsToProposals(proposals, params_.nms_threshold, picked);

            const int model_w = params_.model_in_w;
            const int model_h = params_.model_in_h;

            result_.group.detections.clear();

            for (int idx : picked)
            {
                if (result_.group.detections.size() >= static_cast<std::size_t>(params_.obj_numb_max_size))
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
                const float x1 = std::max(0.f, std::min(obj.x + obj.w, static_cast<float>(model_w)));
                const float y1 = std::max(0.f, std::min(obj.y + obj.h, static_cast<float>(model_h)));
                obj.w = x1 - obj.x;
                obj.h = y1 - obj.y;

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

                result_.group.detections.push_back(det);
            }

            result_.success = !result_.group.detections.empty();
            return result_.success;
        }

        void YoloV8DetectPostProcessAwnn::generateProposals(
            int stride,
            const float *feat_grid,
            const float *feat_score,
            float prob_threshold,
            std::vector<Proposal> &objects)
        {
            const int letterbox_cols = params_.model_in_w;
            const int letterbox_rows = params_.model_in_h;
            const int num_class = params_.obj_class_num;

            const int num_grid_x = letterbox_cols / stride;
            const int num_grid_y = letterbox_rows / stride;
            const int num_grid_size = num_grid_x * num_grid_y;

            const float deprob_threshold = YoloBasePostProcess::desigmoid(prob_threshold);
            float dst[kRegMax]{};

            for (int y = 0; y < num_grid_y; ++y)
            {
                for (int x = 0; x < num_grid_x; ++x)
                {
                    const int num_grid_idx = y * num_grid_x + x;

                    int label = -1;
                    float score = -FLT_MAX;
                    for (int k = 0; k < num_class; ++k)
                    {
                        const float s = feat_score[k * num_grid_size + num_grid_idx];
                        if (s > score)
                        {
                            label = k;
                            score = s;
                        }
                    }

                    if (score < deprob_threshold)
                    {
                        continue;
                    }

                    score = YoloBasePostProcess::sigmoid(score);
                    if (score < prob_threshold)
                    {
                        continue;
                    }

                    float pred_grid[kRegMax * 4]{};
                    const float *cur_pred_grid = feat_grid + num_grid_idx;
                    for (int i = 0; i < kRegMax * 4; ++i)
                    {
                        pred_grid[i] = cur_pred_grid[i * num_grid_size];
                    }

                    const float x0 = (x + 0.5f - softmaxExpectation(pred_grid, dst, kRegMax)) * stride;
                    const float y0 = (y + 0.5f - softmaxExpectation(pred_grid + kRegMax, dst, kRegMax)) * stride;
                    const float x1 = (x + 0.5f + softmaxExpectation(pred_grid + 2 * kRegMax, dst, kRegMax)) * stride;
                    const float y1 = (y + 0.5f + softmaxExpectation(pred_grid + 3 * kRegMax, dst, kRegMax)) * stride;

                    Proposal obj{};
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = label;
                    obj.prob = score;
                    objects.push_back(obj);
                }
            }
        }

    } // namespace post_process
} // namespace deploy_percept
