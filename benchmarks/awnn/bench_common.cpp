#include "bench_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

fs::path percept_root()
{
    if (const char *root = std::getenv("PERCEPT_ROOT"))
    {
        return root;
    }

    const fs::path cwd_share = fs::current_path() / "share" / "percept";
    if (fs::is_directory(cwd_share / "apps"))
    {
        return cwd_share;
    }

    char exe_path[4096];
    const ssize_t len = ::readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0)
    {
        exe_path[len] = '\0';
        const fs::path exe = exe_path;
        const fs::path exe_dir = exe.parent_path();

        if (exe_dir.filename() == "benchmarks")
        {
            const fs::path percept_dir = exe_dir.parent_path();
            if (percept_dir.filename() == "percept")
            {
                return percept_dir;
            }
        }

        const fs::path prefix = exe_dir.parent_path();
        const fs::path install_share = prefix / "share" / "percept";
        if (fs::is_directory(install_share / "apps"))
        {
            return install_share;
        }
    }

    return {};
}

fs::path app_resource(const char *relative)
{
    if (const fs::path root = percept_root(); !root.empty())
    {
        return root / "apps" / "yolov5_detect_awnn" / relative;
    }
    return fs::path("apps/yolov5_detect_awnn") / relative;
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

bool run_preprocess(
    const PreprocessMode mode,
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
    input_buffer = pack_rgb_for_vip_input(resized, model_w, model_h, model_c);
    return true;
}

} // namespace awnn
} // namespace bench
} // namespace percept
