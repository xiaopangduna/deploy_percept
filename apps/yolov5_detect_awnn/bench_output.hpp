#pragma once

#include <cstdint>
#include <vector>

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

namespace yolov5_detect_awnn_bench
{

struct BenchStats
{
    double npu_ms_avg{0};
    double output_fetch_ms_avg{0};
    double post_ms_avg{0};
    double output_plus_post_ms_avg{0};
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

/** Mapped vs HostCopy 两实例检测数量是否一致 */
bool verify_output_paths_match(
    deploy_percept::engine::AwnnEngine &mapped_engine,
    deploy_percept::engine::AwnnEngine &host_copy_engine,
    deploy_percept::post_process::YoloV5DetectPostProcessAwnn &processor,
    const std::vector<std::uint8_t> &input_nchw,
    int model_h,
    int model_w);

} // namespace yolov5_detect_awnn_bench
