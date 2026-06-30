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
    const char *preprocess{nullptr}; ///< none | opencv_mem | opencv_disk
    const char *infer{nullptr};      ///< NPU | CPU
    const char *copy_in{nullptr};    ///< hostcopy | zerocopy
    const char *copy_out{nullptr};   ///< mapped | hostcopy
    const char *post{nullptr};       ///< awnn_mapped | awnn_host
};

struct BenchRowStats
{
    BenchRowConfig cfg{};
    double pre_ms{0};
    double infer_ms{0};
    double copy_ms{0};
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

/** @return false if copy_out/post combination is invalid */
bool validate_bench_row_config(const BenchRowConfig &cfg);

/** true when post phase should memcpy outputs to host before processor.run() */
bool host_copy_before_post(const BenchRowConfig &cfg);

void print_bench_report(const BenchReport &report);
void print_fastest_summary(const BenchReport &report);

} // namespace awnn
} // namespace bench
} // namespace percept
