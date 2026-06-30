#pragma once

#include "bench_report.hpp"

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

namespace percept {
namespace bench {
namespace awnn {

BenchRowStats bench_one_row(
    deploy_percept::engine::AwnnEngine &engine,
    deploy_percept::post_process::YoloV5DetectPostProcessAwnn &processor,
    const std::string &input_path,
    int model_w,
    int model_h,
    int model_c,
    const BenchRowConfig &cfg,
    int warmup,
    int loops);

} // namespace awnn
} // namespace bench
} // namespace percept
