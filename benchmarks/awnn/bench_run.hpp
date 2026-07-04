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
    const char *fixture_case{nullptr};
    const char *default_model{nullptr};
    ModelInputLayout input_layout{ModelInputLayout::YoloV5};
    InputPackLayout pack_layout{InputPackLayout::Chw};
    const char *default_input{"dog.jpg"};
};

struct BenchRunContext
{
    int loops{50};
    std::string model_path;
    std::string input_path;
    int model_w{0};
    int model_h{0};
    int model_c{0};
    InputPackLayout pack_layout{InputPackLayout::Chw};
};

constexpr int kDefaultWarmup = 5;
constexpr int kDefaultLoops = 50;

/** Parse [loops] [model.nb] [input.jpg]; query model input dims. */
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

        deploy_percept::engine::AwnnEngine::Params engine_params;
        engine_params.model_path = ctx.model_path;
        engine_params.is_copy_host = row_cfg.is_copy_host;

        deploy_percept::engine::AwnnEngine engine(engine_params);
        if (!engine.is_valid())
        {
            std::fprintf(
                stderr,
                "AwnnEngine init failed for tag=%s (is_copy_host=%d)\n",
                row_cfg.tag,
                row_cfg.is_copy_host ? 1 : 0);
            continue;
        }

        std::printf(
            "\n--- running tag=%s focus=%s is_copy_host=%d ---\n",
            row_cfg.tag,
            row_cfg.focus,
            row_cfg.is_copy_host ? 1 : 0);

        const BenchRowStats stats = bench_one_row(
            engine,
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
