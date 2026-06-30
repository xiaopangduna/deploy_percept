#include "bench_run.hpp"

#include "bench_common.hpp"
#include "bench_pipeline.hpp"

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/engine/VipLiteRuntime.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace percept {
namespace bench {
namespace awnn {

namespace
{

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::YoloV5DetectPostProcessAwnn;

} // namespace

bool init_bench_run_context(const int argc, char **argv, BenchRunContext &ctx)
{
    const fs::path base = fs::current_path();
    const std::string default_model = app_resource("yolov5.nb").string();
    const std::string default_input = app_resource("dog.jpg").string();

    int loops = kDefaultLoops;
    int argi = 1;
    if (argi < argc && argv[argi][0] >= '0' && argv[argi][0] <= '9')
    {
        loops = std::atoi(argv[argi]);
        ++argi;
    }

    const char *model_arg = (argi < argc) ? argv[argi] : default_model.c_str();
    const char *input_arg = (argi + 1 < argc) ? argv[argi + 1] : default_input.c_str();

    ctx.loops = loops;
    ctx.model_path = resolve_path(model_arg, base);
    ctx.input_path = resolve_path(input_arg, base);

    if (!fs::is_regular_file(ctx.model_path))
    {
        std::fprintf(stderr, "model not found: %s\n", ctx.model_path.c_str());
        return false;
    }
    if (!fs::is_regular_file(ctx.input_path))
    {
        std::fprintf(stderr, "input not found: %s\n", ctx.input_path.c_str());
        return false;
    }

    static VipLiteRuntime runtime;
    if (!runtime.ok())
    {
        std::fprintf(
            stderr,
            "VipLiteRuntime init failed (check LD_LIBRARY_PATH=$PWD/lib)\n");
        return false;
    }

    static AwnnEngine::Param params;
    params.model_path = ctx.model_path;

    static AwnnEngine engine(params);
    if (!engine.is_valid())
    {
        std::fprintf(stderr, "AwnnEngine init failed\n");
        return false;
    }

    const auto &model_info = engine.getInfo();
    const auto &sizes = model_info.input_sizes.at(0);
    ctx.model_w = static_cast<int>(sizes[0]);
    ctx.model_h = static_cast<int>(sizes[1]);
    ctx.model_c = static_cast<int>(sizes[2]);

    static YoloV5DetectPostProcessAwnn::Params post_params;
    post_params.model_in_h = ctx.model_h;
    post_params.model_in_w = ctx.model_w;
    static YoloV5DetectPostProcessAwnn processor(post_params);

    ctx.engine = &engine;
    ctx.processor = &processor;
    return true;
}

BenchReport run_bench_suite(
    const char *title,
    const BenchRunContext &ctx,
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
            *ctx.processor,
            ctx.input_path,
            ctx.model_w,
            ctx.model_h,
            ctx.model_c,
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
