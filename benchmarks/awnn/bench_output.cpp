#include "bench_output.hpp"

#include "deploy_percept/engine/AwnnResultGuard.hpp"
#include "deploy_percept/post_process/types.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace percept {
namespace bench {
namespace awnn {

namespace
{

using deploy_percept::engine::AwnnEngine;
using deploy_percept::post_process::TensorView;

const char *bench_path_label(const bool host_copy_before_post)
{
    return host_copy_before_post ? "HostCopy" : "Mapped";
}

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

BenchStats bench_output_path(
    AwnnEngine &engine,
    deploy_percept::post_process::YoloV5DetectPostProcessAwnn &processor,
    const std::vector<std::uint8_t> &input_nchw,
    const int warmup,
    const int loops,
    const bool host_copy_before_post)
{
    BenchStats stats{};
    const std::size_t input_size = input_nchw.size();
    const int total = warmup + loops;
    double run_sum = 0;
    double post_sum = 0;

    for (int i = 0; i < total; ++i)
    {
        const auto run_t0 = std::chrono::steady_clock::now();
        if (!engine.run(input_nchw.data(), input_size))
        {
            std::fprintf(
                stderr,
                "bench: %s run() failed at iteration %d\n",
                bench_path_label(host_copy_before_post),
                i);
            return stats;
        }
        const auto run_t1 = std::chrono::steady_clock::now();

        const auto post_t0 = std::chrono::steady_clock::now();
        if (host_copy_before_post)
        {
            const std::vector<TensorView> &mapped = engine.getResult().outputs;
            if (!engine.getResult().ready || mapped.empty())
            {
                std::fprintf(stderr, "bench: empty output at iteration %d\n", i);
                engine.releaseResult();
                return stats;
            }

            const std::vector<TensorView> host_views = copy_views_to_host(mapped);
            engine.releaseResult();
            if (!processor.run(host_views))
            {
                std::fprintf(stderr, "bench: post failed at iteration %d\n", i);
                return stats;
            }
        }
        else
        {
            deploy_percept::engine::AwnnResultGuard engine_result_guard(engine);
            if (engine_result_guard.empty() || !processor.run(engine_result_guard.views()))
            {
                std::fprintf(stderr, "bench: post failed at iteration %d\n", i);
                return stats;
            }
        }
        const auto post_t1 = std::chrono::steady_clock::now();

        if (i >= warmup)
        {
            run_sum += std::chrono::duration<double, std::milli>(run_t1 - run_t0).count();
            post_sum += std::chrono::duration<double, std::milli>(post_t1 - post_t0).count();
        }
    }

    if (loops <= 0)
    {
        return stats;
    }

    stats.run_ms_avg = run_sum / loops;
    stats.post_ms_avg = post_sum / loops;
    stats.pipeline_ms_avg = stats.run_ms_avg + stats.post_ms_avg;
    return stats;
}

void print_bench_compare(
    const BenchStats &mapped,
    const BenchStats &host_copy,
    const int warmup,
    const int loops)
{
    std::printf("\n=== AWNN output path benchmark (warmup=%d loops=%d) ===\n", warmup, loops);
    std::printf("(preprocess excluded; HostCopy = memcpy in post phase)\n\n");

    auto print_row = [](const char *label, const BenchStats &s) {
        std::printf(
            "%-10s  run=%6.2f ms  post=%6.2f ms  pipeline=%6.2f ms\n",
            label,
            s.run_ms_avg,
            s.post_ms_avg,
            s.pipeline_ms_avg);
    };

    print_row("Mapped", mapped);
    print_row("HostCopy", host_copy);

    if (host_copy.pipeline_ms_avg > 0)
    {
        const double delta =
            (mapped.pipeline_ms_avg - host_copy.pipeline_ms_avg) /
            host_copy.pipeline_ms_avg * 100.0;
        std::printf(
            "\npipeline: Mapped vs HostCopy %+.1f%% (%s faster)\n",
            delta,
            delta <= 0 ? "Mapped" : "HostCopy");
    }
}

} // namespace awnn
} // namespace bench
} // namespace percept
