/**
 * YOLOv5 AWNN 全流程 benchmark：预处理对比（opencv_mem vs opencv_disk，输出固定 hostcopy）
 *
 * 用法:
 *   bench_yolov5_preprocess [loops] [model.nb] [input.jpg]
 */
#include "bench_report.hpp"
#include "bench_run.hpp"

#include <cstdio>
#include <vector>

using percept::bench::awnn::BenchReport;
using percept::bench::awnn::BenchRowConfig;
using percept::bench::awnn::BenchRunContext;
using percept::bench::awnn::init_bench_run_context;
using percept::bench::awnn::kDefaultWarmup;
using percept::bench::awnn::print_bench_report;
using percept::bench::awnn::print_fastest_summary;
using percept::bench::awnn::run_bench_suite;

namespace {

const BenchRowConfig kRows[] = {
    {
        "mem_hostcopy",
        "baseline",
        "opencv_mem",
        "NPU",
        "hostcopy",
        "hostcopy",
        "awnn_host",
    },
    {
        "disk_hostcopy",
        "preprocess",
        "opencv_disk",
        "NPU",
        "hostcopy",
        "hostcopy",
        "awnn_host",
    },
};

} // namespace

int main(int argc, char **argv)
{
    BenchRunContext ctx;
    if (!init_bench_run_context(argc, argv, ctx))
    {
        return 1;
    }

    std::printf("bench_yolov5_preprocess\n");
    std::printf("  suite: preprocess (full pipeline, copy_out=hostcopy)\n");
    std::printf("  model  : %s\n", ctx.model_path.c_str());
    std::printf("  input  : %s\n", ctx.input_path.c_str());
    std::printf("  warmup : %d  loops: %d\n", kDefaultWarmup, ctx.loops);
    std::printf(
        "  model input VIP sizes: W=%d H=%d C=%d\n",
        ctx.model_w,
        ctx.model_h,
        ctx.model_c);

    const BenchReport report = run_bench_suite(
        "yolov5_detect_awnn full pipeline (preprocess)",
        ctx,
        std::vector<BenchRowConfig>(std::begin(kRows), std::end(kRows)));

    print_bench_report(report);
    print_fastest_summary(report);
    return 0;
}
