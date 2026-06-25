#include "bench_output.hpp"

#include <chrono>
#include <cstdio>
#include <vector>

namespace yolov5_detect_awnn_bench
{

namespace
{

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::OutputFetch;

void release_mapped_outputs_if_needed(AwnnEngine &engine)
{
    if (engine.getParams().output_fetch == OutputFetch::Mapped)
    {
        engine.release_outputs();
    }
}

std::vector<float *> collect_output_ptrs(AwnnEngine &engine)
{
    float **raw = engine.output_buffers_float();
    std::vector<float *> outputs;
    if (raw == nullptr)
    {
        return outputs;
    }
    outputs.reserve(engine.output_count());
    for (std::uint32_t i = 0; i < engine.output_count(); ++i)
    {
        outputs.push_back(raw[i]);
    }
    return outputs;
}

int run_post_and_count(
    AwnnEngine &engine,
    deploy_percept::post_process::YoloV5DetectPostProcessAwnn &processor,
    int model_h,
    int model_w)
{
    const std::vector<float *> outputs = collect_output_ptrs(engine);
    if (outputs.empty())
    {
        return -1;
    }
    if (!processor.run(outputs, model_h, model_w))
    {
        return -1;
    }
    return processor.getResult().group.count;
}

const char *output_fetch_label(const AwnnEngine &engine)
{
    return engine.getParams().output_fetch == OutputFetch::Mapped ? "Mapped" : "HostCopy";
}

} // namespace

bool verify_output_paths_match(
    AwnnEngine &mapped_engine,
    AwnnEngine &host_copy_engine,
    deploy_percept::post_process::YoloV5DetectPostProcessAwnn &processor,
    const std::vector<std::uint8_t> &input_nchw,
    int model_h,
    int model_w)
{
    const std::size_t input_size = input_nchw.size();

    if (!mapped_engine.run(input_nchw.data(), input_size))
    {
        std::fprintf(stderr, "verify: Mapped engine run() failed\n");
        return false;
    }
    const int mapped_count = run_post_and_count(mapped_engine, processor, model_h, model_w);
    release_mapped_outputs_if_needed(mapped_engine);
    if (mapped_count < 0)
    {
        std::fprintf(stderr, "verify: post after Mapped run() failed\n");
        return false;
    }

    if (!host_copy_engine.run(input_nchw.data(), input_size))
    {
        std::fprintf(stderr, "verify: HostCopy engine run() failed\n");
        return false;
    }
    const int copy_count = run_post_and_count(host_copy_engine, processor, model_h, model_w);
    if (copy_count < 0)
    {
        std::fprintf(stderr, "verify: post after HostCopy run() failed\n");
        return false;
    }

    if (mapped_count != copy_count)
    {
        std::fprintf(
            stderr,
            "verify: detection count mismatch Mapped=%d HostCopy=%d\n",
            mapped_count,
            copy_count);
        return false;
    }

    std::printf("verify: Mapped and HostCopy both detect %d objects\n", mapped_count);
    return true;
}

BenchStats bench_output_path(
    AwnnEngine &engine,
    deploy_percept::post_process::YoloV5DetectPostProcessAwnn &processor,
    const std::vector<std::uint8_t> &input_nchw,
    const int model_h,
    const int model_w,
    const int warmup,
    const int loops)
{
    BenchStats stats{};
    const std::size_t input_size = input_nchw.size();
    const int total = warmup + loops;
    double npu_sum = 0;
    double output_sum = 0;
    double post_sum = 0;

    for (int i = 0; i < total; ++i)
    {
        if (!engine.run(input_nchw.data(), input_size))
        {
            std::fprintf(
                stderr,
                "bench: %s run() failed at iteration %d\n",
                output_fetch_label(engine),
                i);
            return stats;
        }

        const auto post_t0 = std::chrono::steady_clock::now();
        const std::vector<float *> outputs = collect_output_ptrs(engine);
        if (outputs.empty() || !processor.run(outputs, model_h, model_w))
        {
            std::fprintf(stderr, "bench: post failed at iteration %d\n", i);
            release_mapped_outputs_if_needed(engine);
            return stats;
        }
        const auto post_t1 = std::chrono::steady_clock::now();

        release_mapped_outputs_if_needed(engine);

        if (i >= warmup)
        {
            const deploy_percept::engine::RunTiming &timing = engine.last_run_timing();
            npu_sum += timing.npu_ms;
            output_sum += timing.output_fetch_ms;
            post_sum += std::chrono::duration<double, std::milli>(post_t1 - post_t0).count();
        }
    }

    if (loops <= 0)
    {
        return stats;
    }

    stats.npu_ms_avg = npu_sum / loops;
    stats.output_fetch_ms_avg = output_sum / loops;
    stats.post_ms_avg = post_sum / loops;
    stats.output_plus_post_ms_avg = stats.output_fetch_ms_avg + stats.post_ms_avg;
    return stats;
}

void print_bench_compare(
    const BenchStats &mapped,
    const BenchStats &host_copy,
    const int warmup,
    const int loops)
{
    std::printf("\n=== AWNN output path benchmark (warmup=%d loops=%d) ===\n", warmup, loops);
    std::printf("(preprocess excluded; input prepared once before bench)\n\n");

    auto print_row = [](const char *label, const BenchStats &s) {
        std::printf(
            "%-10s  npu=%6.2f ms  output_fetch=%6.2f ms  post=%6.2f ms  output+post=%6.2f ms\n",
            label,
            s.npu_ms_avg,
            s.output_fetch_ms_avg,
            s.post_ms_avg,
            s.output_plus_post_ms_avg);
    };

    print_row("Mapped", mapped);
    print_row("HostCopy", host_copy);

    if (host_copy.output_plus_post_ms_avg > 0)
    {
        const double delta =
            (mapped.output_plus_post_ms_avg - host_copy.output_plus_post_ms_avg) /
            host_copy.output_plus_post_ms_avg * 100.0;
        std::printf(
            "\noutput+post: Mapped vs HostCopy %+.1f%% (%s faster)\n",
            delta,
            delta <= 0 ? "Mapped" : "HostCopy");
    }
}

} // namespace yolov5_detect_awnn_bench
