#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace percept {
namespace bench {
namespace awnn {

enum class PreprocessMode
{
    OpenCvMem,  ///< resize + pack from in-memory BGR (no imread per frame)
    OpenCvDisk, ///< imread + resize + pack each frame
};

/** VIP input buffer 线性布局，与对应 demo 一致 */
enum class InputPackLayout
{
    Chw, ///< YOLOv5：[W,H,C,N] + NCHW pack
    Hwc, ///< YOLOv8：[C,H,W,N] + RGB HWC 交错
};

/** VIP `input_sizes` 维度语义（与 apps 内 demo 一致） */
enum class ModelInputLayout
{
    YoloV5, ///< [W, H, C, N]
    YoloV8, ///< [C, H, W, N]
};

PreprocessMode parse_preprocess_mode(const char *preprocess);

std::filesystem::path percept_root();
std::filesystem::path app_resource(const char *app_name, const char *relative);

void parse_model_input_dims(
    const std::array<std::uint32_t, 6> &sizes,
    ModelInputLayout layout,
    int &model_w,
    int &model_h,
    int &model_c);
std::string resolve_path(const char *path, const std::filesystem::path &base);

std::vector<std::uint8_t> pack_rgb_for_vip_input(
    const cv::Mat &bgr,
    int width,
    int height,
    int channels);

/**
 * Run preprocess for one frame.
 * @p cached_bgr  for OpenCvMem: source image already in memory (not timed imread)
 */
bool run_preprocess(
    PreprocessMode mode,
    InputPackLayout pack_layout,
    const std::string &input_path,
    const cv::Mat &cached_bgr,
    int model_w,
    int model_h,
    int model_c,
    std::vector<std::uint8_t> &input_buffer);

} // namespace awnn
} // namespace bench
} // namespace percept
