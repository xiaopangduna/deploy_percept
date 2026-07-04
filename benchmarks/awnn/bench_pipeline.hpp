#pragma once

#include "bench_common.hpp"
#include "bench_report.hpp"

#include "deploy_percept/core/engine/AwnnEngine.hpp"
#include "deploy_percept/core/engine/EngineResultGuard.hpp"

#include <chrono>
#include <cstdio>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace percept {
namespace bench {
namespace awnn {

template <typename PostProcessor>
BenchRowStats bench_one_row(
    deploy_percept::engine::AwnnEngine &engine,
    PostProcessor &processor,
    const std::string &input_path,
    const int model_w,
    const int model_h,
    const int model_c,
    const InputPackLayout pack_layout,
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
    double post_sum = 0;

    std::vector<std::uint8_t> input_buffer;
    input_buffer.reserve(input_size);

    for (int i = 0; i < total; ++i)
    {
        const auto pre_t0 = std::chrono::steady_clock::now();
        if (!run_preprocess(
                preprocess_mode,
                pack_layout,
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

        const auto post_t0 = std::chrono::steady_clock::now();
        bool post_ok = false;
        if (cfg.is_copy_host)
        {
            const auto &tensor_views = engine.getResult().tensor_views;
            if (!tensor_views.empty())
            {
                post_ok = processor.run(tensor_views);
            }
            engine.releaseResult();
        }
        else
        {
            deploy_percept::engine::EngineResultGuard engine_result_guard(engine);
            const auto &tensor_views = engine_result_guard.getEngineResult().tensor_views;
            post_ok = !tensor_views.empty() && processor.run(tensor_views);
        }
        const auto post_t1 = std::chrono::steady_clock::now();

        if (!post_ok)
        {
            std::fprintf(
                stderr,
                "bench: post failed for tag=%s at iteration %d\n",
                cfg.tag,
                i);
            return stats;
        }

        if (i >= warmup)
        {
            pre_sum += std::chrono::duration<double, std::milli>(pre_t1 - pre_t0).count();
            infer_sum += std::chrono::duration<double, std::milli>(infer_t1 - infer_t0).count();
            post_sum += std::chrono::duration<double, std::milli>(post_t1 - post_t0).count();
        }
    }

    if (loops <= 0)
    {
        return stats;
    }

    stats.pre_ms = pre_sum / loops;
    stats.infer_ms = infer_sum / loops;
    stats.post_ms = post_sum / loops;
    stats.pipeline_ms = stats.pre_ms + stats.infer_ms + stats.post_ms;
    return stats;
}

} // namespace awnn
} // namespace bench
} // namespace percept
