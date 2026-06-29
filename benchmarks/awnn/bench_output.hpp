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
    double engine_run_ms_avg{0}; ///< engine.run()
    double copy_ms_avg{0};       ///< 输出 memcpy 到 host（仅 HostCopy 路径）
    double post_ms_avg{0};       ///< processor.run()（不含 copy）
    double pipeline_ms_avg{0};   ///< engine_run + copy + post
};

/** @p host_copy_before_post：true 时在 post 前将 mapped 输出 memcpy 到 host */
BenchStats bench_output_path(
    deploy_percept::engine::AwnnEngine &engine,
    deploy_percept::post_process::YoloV5DetectPostProcessAwnn &processor,
    const std::vector<std::uint8_t> &input_buffer,
    int warmup,
    int loops,
    bool host_copy_before_post);

void print_bench_compare(
    const BenchStats &mapped,
    const BenchStats &host_copy,
    int warmup,
    int loops);

} // namespace awnn
} // namespace bench
} // namespace percept
