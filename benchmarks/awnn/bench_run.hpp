#pragma once

#include "bench_report.hpp"

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

#include <string>
#include <vector>

namespace percept {
namespace bench {
namespace awnn {

struct BenchRunContext
{
    int loops{50};
    std::string model_path;
    std::string input_path;
    deploy_percept::engine::AwnnEngine *engine{nullptr};
    deploy_percept::post_process::YoloV5DetectPostProcessAwnn *processor{nullptr};
    int model_w{0};
    int model_h{0};
    int model_c{0};
};

constexpr int kDefaultWarmup = 5;
constexpr int kDefaultLoops = 50;

/** Parse [loops] [model.nb] [input.jpg]; initialize runtime, engine, processor. */
bool init_bench_run_context(int argc, char **argv, BenchRunContext &ctx);

BenchReport run_bench_suite(
    const char *title,
    const BenchRunContext &ctx,
    const std::vector<BenchRowConfig> &rows);

} // namespace awnn
} // namespace bench
} // namespace percept
