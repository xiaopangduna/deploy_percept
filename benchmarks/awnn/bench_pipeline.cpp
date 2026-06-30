#include "bench_pipeline.hpp"

#include "bench_common.hpp"
#include "bench_report.hpp"

#include "deploy_percept/engine/AwnnResultGuard.hpp"
#include "deploy_percept/post_process/types.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include <opencv2/core.hpp>

namespace percept {
namespace bench {
namespace awnn {

namespace
{

using deploy_percept::engine::AwnnEngine;
using deploy_percept::post_process::TensorView;

std::vector<TensorView> copy_views_to_host(const std::vector<TensorView> &mapped)
{
    static thread_local std::vector<std::vector<std::uint8_t>> host_storage;
    host_storage.clear();
    host_storage.resize(mapped.size());

    std::vector<TensorView> host_views;
    host_views.reserve(mapped.size());

    for (std::size_t i = 0; i < mapped.size(); ++i)
    {
        const TensorView &view = mapped[i];
        TensorView copy{};
        copy.byte_size = view.byte_size;
        copy.dtype = view.dtype;
        if (view.data != nullptr && view.byte_size > 0)
        {
            host_storage[i].resize(view.byte_size);
            std::memcpy(host_storage[i].data(), view.data, view.byte_size);
            copy.data = host_storage[i].data();
        }
        host_views.push_back(copy);
    }

    return host_views;
}

} // namespace

BenchRowStats bench_one_row(
    AwnnEngine &engine,
    deploy_percept::post_process::YoloV5DetectPostProcessAwnn &processor,
    const std::string &input_path,
    const int model_w,
    const int model_h,
    const int model_c,
    const BenchRowConfig &cfg,
    const int warmup,
    const int loops)
{
    BenchRowStats stats{};
    stats.cfg = cfg;

    if (!validate_bench_row_config(cfg))
    {
        std::fprintf(stderr, "bench: invalid config for tag=%s\n", cfg.tag != nullptr ? cfg.tag : "?");
        return stats;
    }

    const PreprocessMode preprocess_mode = parse_preprocess_mode(cfg.preprocess);
    const bool copy_before_post = host_copy_before_post(cfg);

    cv::Mat cached_bgr;
    if (preprocess_mode == PreprocessMode::OpenCvMem)
    {
        cached_bgr = cv::imread(input_path, cv::IMREAD_COLOR);
        if (cached_bgr.empty())
        {
            std::fprintf(stderr, "failed to read input for cached preprocess: %s\n", input_path.c_str());
            return stats;
        }
    }

    const int total = warmup + loops;
    const std::size_t input_size =
        static_cast<std::size_t>(model_w) * static_cast<std::size_t>(model_h) *
        static_cast<std::size_t>(model_c);

    double pre_sum = 0;
    double infer_sum = 0;
    double copy_sum = 0;
    double post_sum = 0;

    std::vector<std::uint8_t> input_buffer;
    input_buffer.reserve(input_size);

    for (int i = 0; i < total; ++i)
    {
        const auto pre_t0 = std::chrono::steady_clock::now();
        if (!run_preprocess(
                preprocess_mode,
                input_path,
                cached_bgr,
                model_w,
                model_h,
                model_c,
                input_buffer))
        {
            std::fprintf(
                stderr,
                "bench: preprocess failed for tag=%s at iteration %d\n",
                cfg.tag,
                i);
            return stats;
        }
        const auto pre_t1 = std::chrono::steady_clock::now();

        const auto infer_t0 = std::chrono::steady_clock::now();
        if (!engine.run(input_buffer.data(), input_buffer.size()))
        {
            std::fprintf(
                stderr,
                "bench: engine.run() failed for tag=%s at iteration %d\n",
                cfg.tag,
                i);
            return stats;
        }
        const auto infer_t1 = std::chrono::steady_clock::now();

        double copy_ms = 0;
        double post_ms = 0;

        if (copy_before_post)
        {
            const std::vector<TensorView> &mapped = engine.getResult().outputs;
            if (!engine.getResult().ready || mapped.empty())
            {
                std::fprintf(
                    stderr,
                    "bench: empty output for tag=%s at iteration %d\n",
                    cfg.tag,
                    i);
                engine.releaseResult();
                return stats;
            }

            const auto copy_t0 = std::chrono::steady_clock::now();
            const std::vector<TensorView> host_views = copy_views_to_host(mapped);
            const auto copy_t1 = std::chrono::steady_clock::now();
            copy_ms = std::chrono::duration<double, std::milli>(copy_t1 - copy_t0).count();

            engine.releaseResult();

            const auto post_t0 = std::chrono::steady_clock::now();
            if (!processor.run(host_views))
            {
                std::fprintf(
                    stderr,
                    "bench: post failed for tag=%s at iteration %d\n",
                    cfg.tag,
                    i);
                return stats;
            }
            const auto post_t1 = std::chrono::steady_clock::now();
            post_ms = std::chrono::duration<double, std::milli>(post_t1 - post_t0).count();
        }
        else
        {
            const auto post_t0 = std::chrono::steady_clock::now();
            deploy_percept::engine::AwnnResultGuard engine_result_guard(engine);
            if (engine_result_guard.empty() || !processor.run(engine_result_guard.views()))
            {
                std::fprintf(
                    stderr,
                    "bench: post failed for tag=%s at iteration %d\n",
                    cfg.tag,
                    i);
                return stats;
            }
            const auto post_t1 = std::chrono::steady_clock::now();
            post_ms = std::chrono::duration<double, std::milli>(post_t1 - post_t0).count();
        }

        if (i >= warmup)
        {
            pre_sum += std::chrono::duration<double, std::milli>(pre_t1 - pre_t0).count();
            infer_sum += std::chrono::duration<double, std::milli>(infer_t1 - infer_t0).count();
            copy_sum += copy_ms;
            post_sum += post_ms;
        }
    }

    if (loops <= 0)
    {
        return stats;
    }

    stats.pre_ms = pre_sum / loops;
    stats.infer_ms = infer_sum / loops;
    stats.copy_ms = copy_sum / loops;
    stats.post_ms = post_sum / loops;
    stats.pipeline_ms = stats.pre_ms + stats.infer_ms + stats.copy_ms + stats.post_ms;
    return stats;
}

} // namespace awnn
} // namespace bench
} // namespace percept
