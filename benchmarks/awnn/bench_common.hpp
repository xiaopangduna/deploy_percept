#pragma once

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

PreprocessMode parse_preprocess_mode(const char *preprocess);

std::filesystem::path percept_root();
std::filesystem::path app_resource(const char *relative);
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
    const std::string &input_path,
    const cv::Mat &cached_bgr,
    int model_w,
    int model_h,
    int model_c,
    std::vector<std::uint8_t> &input_buffer);

} // namespace awnn
} // namespace bench
} // namespace percept
