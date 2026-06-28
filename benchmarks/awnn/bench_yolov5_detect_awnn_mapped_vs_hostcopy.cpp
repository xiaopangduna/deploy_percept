/**
 * AWNN YOLOv5 检测：Mapped（直接读 map）vs HostCopy（post 前 memcpy）性能对比
 *
 * 用法:
 *   bench_yolov5_detect_awnn_mapped_vs_hostcopy [loops] [model.nb] [input.jpg]
 */
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bench_output.hpp"
#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/engine/VipLiteRuntime.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::YoloV5DetectPostProcessAwnn;
using percept::bench::awnn::bench_output_path;
using percept::bench::awnn::print_bench_compare;

namespace {

constexpr int kWarmup = 5;
constexpr int kDefaultLoops = 50;

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

std::vector<std::uint8_t> pack_nchw_rgb_uint8(
    const cv::Mat &bgr,
    const int width,
    const int height,
    const int channels)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::vector<std::uint8_t> nchw(static_cast<std::size_t>(width * height * channels));
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            const cv::Vec3b pix = rgb.at<cv::Vec3b>(h, w);
            for (int c = 0; c < channels; ++c)
            {
                nchw[c * height * width + h * width + w] = pix[c];
            }
        }
    }
    return nchw;
}

bool prepare_input_nchw(
    AwnnEngine &engine,
    const std::string &input_path,
    std::vector<std::uint8_t> &input_nchw)
{
    const auto &model_info = engine.getInfo();
    const int model_c = static_cast<int>(model_info.input_channels.at(0));
    const int model_h = static_cast<int>(model_info.input_heights.at(0));
    const int model_w = static_cast<int>(model_info.input_widths.at(0));

    cv::Mat orig = cv::imread(input_path, cv::IMREAD_COLOR);
    if (orig.empty())
    {
        std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
        return false;
    }

    cv::Mat resized;
    cv::resize(orig, resized, cv::Size(model_w, model_h));
    input_nchw = pack_nchw_rgb_uint8(resized, model_w, model_h, model_c);
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    const fs::path base = fs::current_path();
    const std::string default_model = app_resource("yolov5.nb").string();
    const std::string default_input = app_resource("dog.jpg").string();

    int loops = kDefaultLoops;
    int argi = 1;
    if (argi < argc && argv[argi][0] >= '0' && argv[argi][0] <= '9')
    {
        loops = std::atoi(argv[argi]);
        ++argi;
    }

    const char *model_arg = (argi < argc) ? argv[argi] : default_model.c_str();
    const char *input_arg = (argi + 1 < argc) ? argv[argi + 1] : default_input.c_str();

    const std::string model_path = resolve_path(model_arg, base);
    const std::string input_path = resolve_path(input_arg, base);

    std::printf("bench_yolov5_detect_awnn_mapped_vs_hostcopy\n");
    std::printf("  compare: Mapped vs HostCopy (copy in benchmark post phase)\n");
    std::printf("  model  : %s\n", model_path.c_str());
    std::printf("  input  : %s\n", input_path.c_str());
    std::printf("  warmup : %d  loops: %d\n", kWarmup, loops);

    if (!fs::is_regular_file(model_path))
    {
        std::fprintf(stderr, "model not found: %s\n", model_path.c_str());
        return 1;
    }
    if (!fs::is_regular_file(input_path))
    {
        std::fprintf(stderr, "input not found: %s\n", input_path.c_str());
        return 1;
    }

    VipLiteRuntime runtime;
    if (!runtime.ok())
    {
        std::fprintf(
            stderr,
            "VipLiteRuntime init failed (check LD_LIBRARY_PATH=$PWD/lib)\n");
        return 1;
    }

    AwnnEngine::Param params;
    params.model_path = model_path;
    params.input_channels = 3;
    params.input_height = 640;
    params.input_width = 640;

    AwnnEngine engine(params);
    if (!engine.is_valid())
    {
        std::fprintf(stderr, "AwnnEngine init failed\n");
        return 1;
    }

    const auto &model_info = engine.getInfo();
    const int model_c = static_cast<int>(model_info.input_channels.at(0));
    const int model_h = static_cast<int>(model_info.input_heights.at(0));
    const int model_w = static_cast<int>(model_info.input_widths.at(0));
    std::printf("  model input: %dx%dx%d (C×H×W)\n", model_c, model_h, model_w);

    std::vector<std::uint8_t> input_nchw;
    if (!prepare_input_nchw(engine, input_path, input_nchw))
    {
        return 1;
    }

    YoloV5DetectPostProcessAwnn::Params post_params;
    post_params.model_in_h = model_h;
    post_params.model_in_w = model_w;
    YoloV5DetectPostProcessAwnn processor(post_params);

    std::printf("\noutput bytes per head:");
    for (std::size_t i = 0; i < engine.getInfo().output_byte_sizes.size(); ++i)
    {
        std::printf(" [%zu]=%u", i, engine.getInfo().output_byte_sizes[i]);
    }
    std::printf("\n");

    const percept::bench::awnn::BenchStats mapped_stats =
        bench_output_path(engine, processor, input_nchw, kWarmup, loops, false);
    const percept::bench::awnn::BenchStats copy_stats =
        bench_output_path(engine, processor, input_nchw, kWarmup, loops, true);

    print_bench_compare(mapped_stats, copy_stats, kWarmup, loops);
    return 0;
}
