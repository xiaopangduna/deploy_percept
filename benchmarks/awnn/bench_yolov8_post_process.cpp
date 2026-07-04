/**
 * YOLOv8 AWNN 全流程 benchmark：对比 is_copy_host mapped vs host
 *
 * 用法:
 *   bench_yolov8_post_process [loops] [model.nb] [input.jpg]
 */
#include "bench_report.hpp"
#include "bench_run.hpp"

#include "deploy_percept/core/post_process/YoloV8DetectPostProcessAwnn.hpp"

#include <cstdio>
#include <vector>

using deploy_percept::post_process::YoloV8DetectPostProcessAwnn;
using percept::bench::awnn::BenchReport;
using percept::bench::awnn::BenchRowConfig;
using percept::bench::awnn::BenchRunContext;
using percept::bench::awnn::InputPackLayout;
using percept::bench::awnn::ModelBenchSpec;
using percept::bench::awnn::ModelInputLayout;
using percept::bench::awnn::init_bench_run_context;
using percept::bench::awnn::kDefaultWarmup;
using percept::bench::awnn::print_bench_report;
using percept::bench::awnn::print_fastest_summary;
using percept::bench::awnn::run_bench_suite;

namespace {

const BenchRowConfig kRows[] = {
    {"mapped", "pipeline", "opencv_mem", false},
    {"host", "pipeline", "opencv_mem", true},
};

constexpr ModelBenchSpec kSpec{
    "yolov8_detect",
    "yolov8.nb",
    ModelInputLayout::YoloV8,
    InputPackLayout::Hwc,
};

} // namespace

int main(int argc, char **argv)
{
    BenchRunContext ctx;
    if (!init_bench_run_context(argc, argv, kSpec, ctx))
    {
        return 1;
    }

    YoloV8DetectPostProcessAwnn::Params post_params;
    post_params.model_in_h = ctx.model_h;
    post_params.model_in_w = ctx.model_w;
    YoloV8DetectPostProcessAwnn processor(post_params);

    std::printf("bench_yolov8_post_process\n");
    std::printf("  suite: full pipeline (is_copy_host mapped vs host)\n");
    std::printf("  model  : %s\n", ctx.model_path.c_str());
    std::printf("  input  : %s\n", ctx.input_path.c_str());
    std::printf("  warmup : %d  loops: %d\n", kDefaultWarmup, ctx.loops);
    std::printf(
        "  model input VIP sizes: C=%d H=%d W=%d\n",
        ctx.model_c,
        ctx.model_h,
        ctx.model_w);

    const BenchReport report = run_bench_suite(
        "yolov8_detect_awnn full pipeline (post_process)",
        ctx,
        processor,
        std::vector<BenchRowConfig>(std::begin(kRows), std::end(kRows)));

    print_bench_report(report);
    print_fastest_summary(report);
    return 0;
}
