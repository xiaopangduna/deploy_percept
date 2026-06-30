#include "bench_report.hpp"

#include <cstdio>
#include <cstring>

namespace percept {
namespace bench {
namespace awnn {

namespace
{

bool streq(const char *a, const char *b)
{
    return a != nullptr && b != nullptr && std::strcmp(a, b) == 0;
}

void print_config_cell(const char *value, int width)
{
    std::printf("%-*s  ", width, value != nullptr ? value : "-");
}

void print_ms_cell(double value)
{
    std::printf("%8.2f  ", value);
}

const BenchRowStats *find_baseline_row(const BenchReport &report)
{
    for (const BenchRowStats &row : report.rows)
    {
        if (streq(row.cfg.focus, "baseline"))
        {
            return &row;
        }
    }
    return report.rows.empty() ? nullptr : &report.rows.front();
}

const BenchRowStats *find_fastest_row(const BenchReport &report)
{
    const BenchRowStats *best = nullptr;
    for (const BenchRowStats &row : report.rows)
    {
        if (best == nullptr || row.pipeline_ms < best->pipeline_ms)
        {
            best = &row;
        }
    }
    return best;
}

} // namespace

bool validate_bench_row_config(const BenchRowConfig &cfg)
{
    if (cfg.tag == nullptr || cfg.focus == nullptr || cfg.preprocess == nullptr ||
        cfg.infer == nullptr || cfg.copy_in == nullptr || cfg.copy_out == nullptr ||
        cfg.post == nullptr)
    {
        return false;
    }

    const bool mapped_out = streq(cfg.copy_out, "mapped");
    const bool hostcopy_out = streq(cfg.copy_out, "hostcopy");
    if (!mapped_out && !hostcopy_out)
    {
        return false;
    }

    if (mapped_out && !streq(cfg.post, "awnn_mapped"))
    {
        return false;
    }
    if (hostcopy_out && !streq(cfg.post, "awnn_host"))
    {
        return false;
    }

    const bool mem = streq(cfg.preprocess, "opencv_mem");
    const bool disk = streq(cfg.preprocess, "opencv_disk");
    return mem || disk;
}

bool host_copy_before_post(const BenchRowConfig &cfg)
{
    return streq(cfg.copy_out, "hostcopy");
}

void print_bench_report(const BenchReport &report)
{
    std::printf("\n=== %s ===\n", report.title != nullptr ? report.title : "benchmark");
    std::printf(
        "model=%s  input=%s  warmup=%d  loops=%d\n",
        report.model_path != nullptr ? report.model_path : "-",
        report.input_path != nullptr ? report.input_path : "-",
        report.warmup,
        report.loops);
    std::printf("pipeline = pre + infer + copy + post (ms/frame)\n\n");

    std::printf(
        "%-14s  %-14s  %-11s  %-5s  %-8s  %-9s  %-11s  "
        "%8s  %8s  %8s  %8s  %8s\n",
        "tag",
        "focus",
        "preprocess",
        "infer",
        "copy_in",
        "copy_out",
        "post",
        "pre",
        "infer",
        "copy",
        "post",
        "pipeline");

    for (const BenchRowStats &row : report.rows)
    {
        const BenchRowConfig &cfg = row.cfg;
        std::printf("%-14s  ", cfg.tag);
        std::printf("%-14s  ", cfg.focus);
        print_config_cell(cfg.preprocess, 11);
        print_config_cell(cfg.infer, 5);
        print_config_cell(cfg.copy_in, 8);
        print_config_cell(cfg.copy_out, 9);
        print_config_cell(cfg.post, 11);
        print_ms_cell(row.pre_ms);
        print_ms_cell(row.infer_ms);
        print_ms_cell(row.copy_ms);
        print_ms_cell(row.post_ms);
        print_ms_cell(row.pipeline_ms);
        std::printf("\n");
    }
}

void print_fastest_summary(const BenchReport &report)
{
    const BenchRowStats *fastest = find_fastest_row(report);
    const BenchRowStats *baseline = find_baseline_row(report);
    if (fastest == nullptr)
    {
        return;
    }

    const double fps = fastest->pipeline_ms > 0.0 ? 1000.0 / fastest->pipeline_ms : 0.0;
    std::printf(
        "\n>>> fastest pipeline: %s (%.2f ms, ~%.1f fps)\n",
        fastest->cfg.tag,
        fastest->pipeline_ms,
        fps);

    if (baseline != nullptr && baseline != fastest && baseline->pipeline_ms > 0.0)
    {
        const double delta =
            (baseline->pipeline_ms - fastest->pipeline_ms) / baseline->pipeline_ms * 100.0;
        std::printf(
            ">>> vs %s: %+.1f%% pipeline (baseline %.2f ms -> fastest %.2f ms)\n",
            baseline->cfg.tag,
            delta,
            baseline->pipeline_ms,
            fastest->pipeline_ms);
    }
}

} // namespace awnn
} // namespace bench
} // namespace percept
