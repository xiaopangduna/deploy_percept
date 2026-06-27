#pragma once

#include <cstdint>
#include <vector>

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

namespace percept {
namespace bench {
namespace awnn {

struct BenchStats
{
    double run_ms_avg{0};       ///< engine.run()：输入 copy + 推理 + 输出取数
    double post_ms_avg{0};      ///< OutputAccess + 后处理
    double pipeline_ms_avg{0};  ///< run + post
};

/** engine 的 Params::output_fetch 决定取数路径 */
BenchStats bench_output_path(
    deploy_percept::engine::AwnnEngine &engine,
    deploy_percept::post_process::YoloV5DetectPostProcessAwnn &processor,
    const std::vector<std::uint8_t> &input_nchw,
    int model_h,
    int model_w,
    int warmup,
    int loops);

void print_bench_compare(
    const BenchStats &mapped,
    const BenchStats &host_copy,
    int warmup,
    int loops);

} // namespace awnn
} // namespace bench
} // namespace percept
