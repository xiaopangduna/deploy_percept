#include "bench_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "deploy_percept/utils/fixture_path.hpp"

namespace fs = std::filesystem;

namespace percept {
namespace bench {
namespace awnn {

PreprocessMode parse_preprocess_mode(const char *preprocess)
{
    if (preprocess != nullptr && std::strcmp(preprocess, "opencv_disk") == 0)
    {
        return PreprocessMode::OpenCvDisk;
    }
    return PreprocessMode::OpenCvMem;
}

fs::path fixture_resource(const char *case_name, const char *relative)
{
    return deploy_percept::utils::fixture_path(
        std::string("integration/awnn/") + case_name + "/" + relative);
}

void parse_model_input_dims(
    const std::array<std::uint32_t, 6> &sizes,
    const ModelInputLayout layout,
    int &model_w,
    int &model_h,
    int &model_c)
{
    if (layout == ModelInputLayout::YoloV8)
    {
        model_c = static_cast<int>(sizes[0]);
        model_h = static_cast<int>(sizes[1]);
        model_w = static_cast<int>(sizes[2]);
        return;
    }

    model_w = static_cast<int>(sizes[0]);
    model_h = static_cast<int>(sizes[1]);
    model_c = static_cast<int>(sizes[2]);
}

std::string resolve_path(const char *path, const fs::path &base)
{
    fs::path p(path);
    if (p.is_relative())
    {
        p = base / p;
    }
    std::error_code ec;
    const fs::path canonical = fs::weakly_canonical(p, ec);
    return ec ? p.string() : canonical.string();
}

std::vector<std::uint8_t> pack_rgb_for_vip_input(
    const cv::Mat &bgr,
    const int width,
    const int height,
    const int channels)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(width * height * channels));
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            const cv::Vec3b pix = rgb.at<cv::Vec3b>(h, w);
            for (int c = 0; c < channels; ++c)
            {
                buffer[c * height * width + h * width + w] = pix[c];
            }
        }
    }
    return buffer;
}

std::vector<std::uint8_t> pack_rgb_hwc_for_vip_input(
    const cv::Mat &bgr,
    const int width,
    const int height)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(width * height * 3));
    cv::Mat rgb_hwc(height, width, CV_8UC3, buffer.data());
    rgb.copyTo(rgb_hwc);
    return buffer;
}

bool run_preprocess(
    const PreprocessMode mode,
    const InputPackLayout pack_layout,
    const std::string &input_path,
    const cv::Mat &cached_bgr,
    const int model_w,
    const int model_h,
    const int model_c,
    std::vector<std::uint8_t> &input_buffer)
{
    cv::Mat source;
    if (mode == PreprocessMode::OpenCvDisk)
    {
        source = cv::imread(input_path, cv::IMREAD_COLOR);
        if (source.empty())
        {
            std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
            return false;
        }
    }
    else
    {
        if (cached_bgr.empty())
        {
            std::fprintf(stderr, "cached input image is empty\n");
            return false;
        }
        source = cached_bgr;
    }

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(model_w, model_h));
    if (pack_layout == InputPackLayout::Hwc)
    {
        input_buffer = pack_rgb_hwc_for_vip_input(resized, model_w, model_h);
    }
    else
    {
        input_buffer = pack_rgb_for_vip_input(resized, model_w, model_h, model_c);
    }
    return true;
}

} // namespace awnn
} // namespace bench
} // namespace percept
