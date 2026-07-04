#pragma once

#include <cstddef>
#include <vector>

namespace percept {
namespace bench {
namespace awnn {

struct BenchRowConfig
{
    const char *tag{nullptr};
    const char *focus{nullptr};
    const char *preprocess{nullptr}; ///< opencv_mem | opencv_disk
    /** 对应 AwnnEngine::Params::is_copy_host */
    bool is_copy_host{true};
};

struct BenchRowStats
{
    BenchRowConfig cfg{};
    double pre_ms{0};
    double infer_ms{0};
    double post_ms{0};
    double pipeline_ms{0};
};

struct BenchReport
{
    const char *title{nullptr};
    const char *model_path{nullptr};
    const char *input_path{nullptr};
    int warmup{0};
    int loops{0};
    std::vector<BenchRowStats> rows;
};

bool validate_bench_row_config(const BenchRowConfig &cfg);

void print_bench_report(const BenchReport &report);
void print_fastest_summary(const BenchReport &report);

} // namespace awnn
} // namespace bench
} // namespace percept
