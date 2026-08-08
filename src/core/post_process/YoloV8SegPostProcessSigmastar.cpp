#include "deploy_percept/core/post_process/YoloV8SegPostProcessSigmastar.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace deploy_percept
{
    namespace post_process
    {

        namespace
        {

            std::uint32_t align_up(std::uint32_t x, std::uint32_t align)
            {
                return (x + align - 1u) & ~(align - 1u);
            }

            void resize_label_mask_nearest(
                const std::vector<std::uint8_t> &src,
                int src_w,
                int src_h,
                std::vector<std::uint8_t> &dst,
                int dst_w,
                int dst_h)
            {
                dst.assign(static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h), 0);
                if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 ||
                    src.size() < static_cast<std::size_t>(src_w) * static_cast<std::size_t>(src_h))
                {
                    return;
                }
                for (int y = 0; y < dst_h; ++y)
                {
                    const int sy = std::min(src_h - 1, (y * src_h) / dst_h);
                    for (int x = 0; x < dst_w; ++x)
                    {
                        const int sx = std::min(src_w - 1, (x * src_w) / dst_w);
                        dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w) +
                            static_cast<std::size_t>(x)] =
                            src[static_cast<std::size_t>(sy) * static_cast<std::size_t>(src_w) +
                                static_cast<std::size_t>(sx)];
                    }
                }
            }

        } // namespace

        YoloV8SegPostProcessSigmastar::YoloV8SegPostProcessSigmastar(const Params &params)
            : params_(params)
        {
        }

        bool YoloV8SegPostProcessSigmastar::run(const std::vector<TensorView> &inputs)
        {
            result_ = Result{};

            if (params_.model_in_h <= 0 || params_.model_in_w <= 0 || params_.num_classes <= 0)
            {
                return false;
            }
            if (inputs.size() != 2 ||
                params_.output_shapes.size() != 2 ||
                params_.output_scalars.size() != 2)
            {
                std::fprintf(
                    stderr,
                    "YoloV8SegPostProcessSigmastar: expect 2 outputs "
                    "(views=%zu shapes=%zu scalars=%zu)\n",
                    inputs.size(),
                    params_.output_shapes.size(),
                    params_.output_scalars.size());
                return false;
            }

            const auto &det_shape = params_.output_shapes[0];
            const auto &proto_shape = params_.output_shapes[1];
            const int channels = static_cast<int>(det_shape[1]);
            const int mask_c = channels - 4 - params_.num_classes;
            const int mask_h = static_cast<int>(proto_shape[1]);
            const int mask_w = static_cast<int>(proto_shape[2]);
            if (mask_c <= 0 || mask_c > kMaxMaskProtosChannel || mask_h <= 0 || mask_w <= 0)
            {
                std::fprintf(
                    stderr,
                    "YoloV8SegPostProcessSigmastar: invalid mask meta "
                    "(C=%d maskC=%d HxW=%dx%d)\n",
                    channels,
                    mask_c,
                    mask_h,
                    mask_w);
                return false;
            }

            std::vector<SegProposal> proposals;
            if (!decodeFusedSeg(inputs[0], proposals))
            {
                return false;
            }
            applyClassAwareNms(proposals);

            // Cap + clamp boxes → detections; keep aligned proposals for mask
            std::vector<SegProposal> kept;
            kept.reserve(proposals.size());
            const int model_w = params_.model_in_w;
            const int model_h = params_.model_in_h;

            for (const auto &obj : proposals)
            {
                if (kept.size() >= static_cast<std::size_t>(params_.obj_numb_max_size))
                {
                    break;
                }
                if (obj.w <= 0.f || obj.h <= 0.f)
                {
                    continue;
                }

                SegProposal clamped = obj;
                clamped.x = std::max(0.f, std::min(clamped.x, static_cast<float>(model_w - 1)));
                clamped.y = std::max(0.f, std::min(clamped.y, static_cast<float>(model_h - 1)));
                const float x2 =
                    std::max(0.f, std::min(clamped.x + clamped.w, static_cast<float>(model_w)));
                const float y2 =
                    std::max(0.f, std::min(clamped.y + clamped.h, static_cast<float>(model_h)));
                clamped.w = x2 - clamped.x;
                clamped.h = y2 - clamped.y;
                if (clamped.w <= 0.f || clamped.h <= 0.f)
                {
                    continue;
                }

                Detection det{};
                det.box.left = static_cast<int>(std::floor(clamped.x));
                det.box.top = static_cast<int>(std::floor(clamped.y));
                det.box.right = static_cast<int>(std::ceil(clamped.x + clamped.w));
                det.box.bottom = static_cast<int>(std::ceil(clamped.y + clamped.h));
                det.score = clamped.prob;
                det.class_id = clamped.label;
                if (det.box.right <= det.box.left || det.box.bottom <= det.box.top)
                {
                    continue;
                }
                result_.detections.push_back(det);
                kept.push_back(clamped);
            }

            if (kept.empty())
            {
                result_.label_mask.assign(
                    static_cast<std::size_t>(model_w) * static_cast<std::size_t>(model_h),
                    0);
                return true;
            }

            return buildMasks(inputs[1], kept, mask_c, mask_h, mask_w);
        }

        bool YoloV8SegPostProcessSigmastar::decodeFusedSeg(
            const TensorView &view,
            std::vector<SegProposal> &proposals)
        {
            if (view.data == nullptr)
            {
                return false;
            }

            const auto &shape = params_.output_shapes[0];
            const std::uint32_t channels = shape[1];
            const std::uint32_t num_boxes = shape[2];
            if (channels < 5 || num_boxes == 0)
            {
                return false;
            }

            const int num_classes = params_.num_classes;
            const int mask_c = static_cast<int>(channels) - 4 - num_classes;
            if (mask_c <= 0 || mask_c > kMaxMaskProtosChannel)
            {
                return false;
            }

            const std::uint32_t stride = align_up(num_boxes, 1u);
            const float scalar = params_.output_scalars[0];

            auto push_box = [&](float cx, float cy, float w, float h, float score, int label,
                                const float *mask_w) {
                SegProposal p{};
                p.x = cx - w / 2.f;
                p.y = cy - h / 2.f;
                p.w = w;
                p.h = h;
                p.prob = score;
                p.label = label;
                for (int c = 0; c < mask_c; ++c)
                {
                    p.mask_weights[static_cast<std::size_t>(c)] = mask_w[static_cast<std::size_t>(c)];
                }
                if (p.w > 0.f && p.h > 0.f)
                {
                    proposals.push_back(p);
                }
            };

            if (view.dtype == TensorDtype::FP32)
            {
                const float *base = static_cast<const float *>(view.data);
                const float *cx_head = base;
                const float *cy_head = base + 1 * stride;
                const float *w_head = base + 2 * stride;
                const float *h_head = base + 3 * stride;

                for (std::uint32_t i = 0; i < num_boxes && proposals.size() < 255; ++i)
                {
                    const float *score_head = base + 4 * stride + i;
                    float max_score = -std::numeric_limits<float>::infinity();
                    int max_ind = -1;
                    for (int c = 0; c < num_classes; ++c)
                    {
                        const float s = score_head[c * static_cast<int>(stride)];
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

                    std::array<float, kMaxMaskProtosChannel> mw{};
                    const float *mask_head =
                        base + static_cast<std::size_t>(4 + num_classes) * stride + i;
                    for (int c = 0; c < mask_c; ++c)
                    {
                        mw[static_cast<std::size_t>(c)] = mask_head[c * static_cast<int>(stride)];
                    }
                    push_box(
                        cx_head[i],
                        cy_head[i],
                        w_head[i],
                        h_head[i],
                        max_score,
                        max_ind,
                        mw.data());
                }
                return true;
            }

            if (view.dtype == TensorDtype::INT16)
            {
                const std::int16_t *base = static_cast<const std::int16_t *>(view.data);
                const std::int16_t *cx_head = base;
                const std::int16_t *cy_head = base + 1 * stride;
                const std::int16_t *w_head = base + 2 * stride;
                const std::int16_t *h_head = base + 3 * stride;
                const float thr_q =
                    (scalar != 0.f) ? (params_.conf_threshold / scalar) : params_.conf_threshold;

                for (std::uint32_t i = 0; i < num_boxes; ++i)
                {
                    const std::int16_t *score_head = base + 4 * stride + i;
                    float best_q = (scalar > 0.f) ? -std::numeric_limits<float>::infinity()
                                                  : std::numeric_limits<float>::infinity();
                    int best_ind = -1;

                    for (int c = 0; c < num_classes; ++c)
                    {
                        const float q =
                            static_cast<float>(score_head[c * static_cast<int>(stride)]);
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

                    std::array<float, kMaxMaskProtosChannel> mw{};
                    const std::int16_t *mask_head =
                        base + static_cast<std::size_t>(4 + num_classes) * stride + i;
                    for (int c = 0; c < mask_c; ++c)
                    {
                        mw[static_cast<std::size_t>(c)] =
                            static_cast<float>(mask_head[c * static_cast<int>(stride)]) * scalar;
                    }
                    push_box(
                        cx_head[i] * scalar,
                        cy_head[i] * scalar,
                        w_head[i] * scalar,
                        h_head[i] * scalar,
                        best_q * scalar,
                        best_ind,
                        mw.data());
                }
                return true;
            }

            std::fprintf(stderr, "YoloV8SegPostProcessSigmastar: fused dtype unsupported\n");
            return false;
        }

        void YoloV8SegPostProcessSigmastar::applyClassAwareNms(
            std::vector<SegProposal> &proposals) const
        {
            if (proposals.size() <= 1)
            {
                return;
            }

            std::sort(
                proposals.begin(),
                proposals.end(),
                [](const SegProposal &a, const SegProposal &b) { return a.prob > b.prob; });

            std::vector<float> areas(proposals.size());
            for (std::size_t i = 0; i < proposals.size(); ++i)
            {
                // OpenDLA：(+1) 面积约定
                areas[i] = (proposals[i].w + 1.f) * (proposals[i].h + 1.f);
            }

            std::vector<SegProposal> kept;
            kept.reserve(proposals.size());
            std::vector<float> kept_areas;
            kept_areas.reserve(proposals.size());

            for (std::size_t i = 0; i < proposals.size(); ++i)
            {
                const auto &a = proposals[i];
                bool suppress = false;
                for (std::size_t k = 0; k < kept.size(); ++k)
                {
                    const auto &b = kept[k];
                    if (a.label != b.label)
                    {
                        continue;
                    }
                    const float x1 = std::max(a.x, b.x);
                    const float y1 = std::max(a.y, b.y);
                    const float x2 = std::min(a.x + a.w, b.x + b.w);
                    const float y2 = std::min(a.y + a.h, b.y + b.h);
                    const float w = std::max(0.f, x2 - x1 + 1.f);
                    const float h = std::max(0.f, y2 - y1 + 1.f);
                    const float inter = w * h;
                    const float uni = areas[i] + kept_areas[k] - inter;
                    if (uni > 0.f && inter / uni >= params_.nms_threshold)
                    {
                        suppress = true;
                        break;
                    }
                }
                if (!suppress)
                {
                    kept.push_back(a);
                    kept_areas.push_back(areas[i]);
                }
            }
            proposals.swap(kept);
        }

        bool YoloV8SegPostProcessSigmastar::buildMasks(
            const TensorView &proto_view,
            const std::vector<SegProposal> &kept,
            int mask_c,
            int mask_h,
            int mask_w)
        {
            if (proto_view.data == nullptr)
            {
                return false;
            }

            const float mask_thr = std::min(std::max(params_.mask_threshold, 1e-6f), 1.f - 1e-6f);
            const float thr_logit = std::log(mask_thr / (1.f - mask_thr));
            const std::uint32_t proto_c_stride = align_up(static_cast<std::uint32_t>(mask_c), 1u);
            const std::uint32_t proto_row_stride =
                static_cast<std::uint32_t>(mask_w) * proto_c_stride;
            const float mask_x_scale =
                static_cast<float>(mask_w) / static_cast<float>(params_.model_in_w);
            const float mask_y_scale =
                static_cast<float>(mask_h) / static_cast<float>(params_.model_in_h);
            const float proto_scalar = params_.output_scalars[1];

            std::vector<std::uint8_t> proto_label(
                static_cast<std::size_t>(mask_w) * static_cast<std::size_t>(mask_h),
                0);
            std::vector<float> float_mask(
                static_cast<std::size_t>(mask_w) * static_cast<std::size_t>(mask_h),
                0.f);

            auto process_roi = [&](const SegProposal &box, auto accumulate_pixel) {
                // 每实例清零，避免 OpenDLA floatMask 跨 box 累积的问题
                std::fill(float_mask.begin(), float_mask.end(), 0.f);

                int xmin = static_cast<int>(box.x * mask_x_scale + 0.5f);
                int xmax = static_cast<int>((box.x + box.w) * mask_x_scale + 0.5f);
                int ymin = static_cast<int>(box.y * mask_y_scale + 0.5f);
                int ymax = static_cast<int>((box.y + box.h) * mask_y_scale + 0.5f);
                xmin = std::min(std::max(0, xmin), mask_w);
                xmax = std::min(std::max(xmin, xmax), mask_w);
                ymin = std::min(std::max(0, ymin), mask_h);
                ymax = std::min(std::max(ymin, ymax), mask_h);

                for (int y = ymin; y < ymax; ++y)
                {
                    for (int x = xmin; x < xmax; ++x)
                    {
                        float_mask[static_cast<std::size_t>(y) * static_cast<std::size_t>(mask_w) +
                                   static_cast<std::size_t>(x)] =
                            accumulate_pixel(y, x, box);
                    }
                }

                const std::uint8_t label_v =
                    static_cast<std::uint8_t>(std::min(255, box.label + 1));
                for (int y = ymin; y < ymax; ++y)
                {
                    for (int x = xmin; x < xmax; ++x)
                    {
                        const std::size_t idx =
                            static_cast<std::size_t>(y) * static_cast<std::size_t>(mask_w) +
                            static_cast<std::size_t>(x);
                        if (float_mask[idx] >= thr_logit)
                        {
                            proto_label[idx] = label_v;
                        }
                    }
                }
            };

            if (proto_view.dtype == TensorDtype::FP32)
            {
                const float *proto = static_cast<const float *>(proto_view.data);
                for (const auto &box : kept)
                {
                    process_roi(box, [&](int y, int x, const SegProposal &b) {
                        const float *p =
                            proto + static_cast<std::size_t>(y) * proto_row_stride +
                            static_cast<std::size_t>(x) * proto_c_stride;
                        float sum = 0.f;
                        for (int c = 0; c < mask_c; ++c)
                        {
                            sum += b.mask_weights[static_cast<std::size_t>(c)] * p[c];
                        }
                        return sum;
                    });
                }
            }
            else if (proto_view.dtype == TensorDtype::INT16)
            {
                const std::int16_t *proto = static_cast<const std::int16_t *>(proto_view.data);
                for (const auto &box : kept)
                {
                    process_roi(box, [&](int y, int x, const SegProposal &b) {
                        const std::int16_t *p =
                            proto + static_cast<std::size_t>(y) * proto_row_stride +
                            static_cast<std::size_t>(x) * proto_c_stride;
                        float sum = 0.f;
                        for (int c = 0; c < mask_c; ++c)
                        {
                            sum += b.mask_weights[static_cast<std::size_t>(c)] *
                                   static_cast<float>(p[c]) * proto_scalar;
                        }
                        return sum;
                    });
                }
            }
            else
            {
                std::fprintf(stderr, "YoloV8SegPostProcessSigmastar: proto dtype unsupported\n");
                return false;
            }

            resize_label_mask_nearest(
                proto_label,
                mask_w,
                mask_h,
                result_.label_mask,
                params_.model_in_w,
                params_.model_in_h);
            return true;
        }

    } // namespace post_process
} // namespace deploy_percept
