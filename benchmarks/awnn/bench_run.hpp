#pragma once

#include "bench_common.hpp"
#include "bench_pipeline.hpp"
#include "bench_report.hpp"

#include "deploy_percept/core/engine/AwnnEngine.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace percept {
namespace bench {
namespace awnn {

struct ModelBenchSpec
{
    const char *app_name{nullptr};
    const char *default_model{nullptr};
    ModelInputLayout input_layout{ModelInputLayout::YoloV5};
    InputPackLayout pack_layout{InputPackLayout::Chw};
};

struct BenchRunContext
{
    int loops{50};
    std::string model_path;
    std::string input_path;
    deploy_percept::engine::AwnnEngine *engine{nullptr};
    int model_w{0};
    int model_h{0};
    int model_c{0};
    InputPackLayout pack_layout{InputPackLayout::Chw};
};

constexpr int kDefaultWarmup = 5;
constexpr int kDefaultLoops = 50;

/** Parse [loops] [model.nb] [input.jpg]; initialize runtime and engine. */
bool init_bench_run_context(
    int argc,
    char **argv,
    const ModelBenchSpec &spec,
    BenchRunContext &ctx);

template <typename PostProcessor>
BenchReport run_bench_suite(
    const char *title,
    const BenchRunContext &ctx,
    PostProcessor &processor,
    const std::vector<BenchRowConfig> &rows)
{
    BenchReport report{};
    report.title = title;
    report.model_path = ctx.model_path.c_str();
    report.input_path = ctx.input_path.c_str();
    report.warmup = kDefaultWarmup;
    report.loops = ctx.loops;
    report.rows.reserve(rows.size());

    for (const BenchRowConfig &row_cfg : rows)
    {
        if (!validate_bench_row_config(row_cfg))
        {
            std::fprintf(stderr, "skip invalid row tag=%s\n", row_cfg.tag != nullptr ? row_cfg.tag : "?");
            continue;
        }

        std::printf("\n--- running tag=%s focus=%s ---\n", row_cfg.tag, row_cfg.focus);
        const BenchRowStats stats = bench_one_row(
            *ctx.engine,
            processor,
            ctx.input_path,
            ctx.model_w,
            ctx.model_h,
            ctx.model_c,
            ctx.pack_layout,
            row_cfg,
            kDefaultWarmup,
            ctx.loops);
        report.rows.push_back(stats);
    }

    return report;
}

} // namespace awnn
} // namespace bench
} // namespace percept
