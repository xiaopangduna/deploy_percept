/**
 * MobileNetV2 AWNN 全流程 benchmark：后处理路径对比（opencv_mem + mapped vs hostcopy）
 *
 * 用法:
 *   bench_mobilenetv2_classify [loops] [model.nb] [input.jpg]
 */
#include "bench_report.hpp"
#include "bench_run.hpp"

#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"

#include <cstdio>
#include <vector>

using deploy_percept::post_process::ClassificationPostProcessAwnn;
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
    {
        "out_mapped",
        "copy_out+post",
        "opencv_mem",
        "NPU",
        "hostcopy",
        "mapped",
        "awnn_mapped",
    },
    {
        "out_hostcopy",
        "copy_out+post",
        "opencv_mem",
        "NPU",
        "hostcopy",
        "hostcopy",
        "awnn_host",
    },
};

constexpr ModelBenchSpec kSpec{
    "mobilenetv2_classify_awnn",
    "mobilenetv2.nb",
    ModelInputLayout::YoloV8,
    InputPackLayout::Hwc,
    "cat.jpg",
};

} // namespace

int main(int argc, char **argv)
{
    BenchRunContext ctx;
    if (!init_bench_run_context(argc, argv, kSpec, ctx))
    {
        return 1;
    }

    ClassificationPostProcessAwnn::Params post_params;
    post_params.top_k = 5;
    ClassificationPostProcessAwnn processor(post_params);

    std::printf("bench_mobilenetv2_classify\n");
    std::printf("  suite: post_process (full pipeline, focus=copy_out+post)\n");
    std::printf("  model  : %s\n", ctx.model_path.c_str());
    std::printf("  input  : %s\n", ctx.input_path.c_str());
    std::printf("  warmup : %d  loops: %d\n", kDefaultWarmup, ctx.loops);
    std::printf(
        "  model input VIP sizes: C=%d H=%d W=%d\n",
        ctx.model_c,
        ctx.model_h,
        ctx.model_w);

    const BenchReport report = run_bench_suite(
        "mobilenetv2_classify_awnn full pipeline (post_process)",
        ctx,
        processor,
        std::vector<BenchRowConfig>(std::begin(kRows), std::end(kRows)));

    print_bench_report(report);
    print_fastest_summary(report);
    return 0;
}
