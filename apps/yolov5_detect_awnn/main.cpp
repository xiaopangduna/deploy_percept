/**
 * Allwinner YOLOv5 检测 demo（AwnnEngine + YoloV5DetectPostProcessAwnn）
 *
 * 默认资源：apps/yolov5_detect_awnn/ → share/percept/apps/yolov5_detect_awnn/
 * 用法：
 *   yolov5_detect_awnn [model.nb] [input.jpg] [output.jpg]
 *   yolov5_detect_awnn --bench [loops] [model.nb] [input.jpg]
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bench_output.hpp"
#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

namespace fs = std::filesystem;
using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::OutputFetch;
using deploy_percept::post_process::YoloV5DetectPostProcessAwnn;

namespace
{

constexpr int kBenchWarmup = 5;
constexpr int kBenchDefaultLoops = 50;

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
        const fs::path prefix = fs::path(exe_path).parent_path().parent_path();
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

bool file_exists(const std::string &path)
{
    return fs::is_regular_file(path);
}

std::vector<uint8_t> pack_nchw_rgb_uint8(const cv::Mat &bgr, int width, int height, int channels)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::vector<uint8_t> nchw(static_cast<std::size_t>(width * height * channels));
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

std::vector<float *> collect_output_ptrs(AwnnEngine &engine)
{
    float **raw = engine.output_buffers_float();
    std::vector<float *> outputs;
    if (raw == nullptr)
    {
        return outputs;
    }
    outputs.reserve(engine.output_count());
    for (std::uint32_t i = 0; i < engine.output_count(); ++i)
    {
        outputs.push_back(raw[i]);
    }
    return outputs;
}

bool prepare_input_nchw(
    AwnnEngine &engine,
    const std::string &input_path,
    cv::Mat &model_input,
    std::vector<uint8_t> &input_nchw)
{
    const int model_w = static_cast<int>(engine.input_width());
    const int model_h = static_cast<int>(engine.input_height());
    const int model_c = static_cast<int>(engine.input_channels());

    cv::Mat orig = cv::imread(input_path, cv::IMREAD_COLOR);
    if (orig.empty())
    {
        std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
        return false;
    }

    cv::resize(orig, model_input, cv::Size(model_w, model_h));
    input_nchw = pack_nchw_rgb_uint8(model_input, model_w, model_h, model_c);
    return true;
}

int run_demo(
    AwnnEngine &engine,
    const std::vector<uint8_t> &input_nchw,
    cv::Mat &model_input,
    const std::string &output_path,
    const int model_h,
    const int model_w)
{
    if (!engine.run(input_nchw.data(), input_nchw.size()))
    {
        std::fprintf(stderr, "AwnnEngine::run failed\n");
        return 1;
    }

    const std::vector<float *> output_buffers_fp32 = collect_output_ptrs(engine);
    if (output_buffers_fp32.empty())
    {
        std::fprintf(stderr, "AwnnEngine output buffers unavailable\n");
        return 1;
    }

    YoloV5DetectPostProcessAwnn::Params post_params;
    YoloV5DetectPostProcessAwnn processor(post_params);
    if (!processor.run(output_buffers_fp32, model_h, model_w))
    {
        std::fprintf(stderr, "post process failed: %s\n", processor.getResult().message.c_str());
        return 1;
    }

    cv::Mat result_img = model_input.clone();
    processor.drawDetectionResults(result_img, processor.getResult().group);

    std::error_code ec;
    fs::create_directories(fs::path(output_path).parent_path(), ec);
    if (!cv::imwrite(output_path, result_img))
    {
        std::fprintf(stderr, "failed to save output to %s\n", output_path.c_str());
        return 1;
    }

    std::printf("detection num: %d\n", processor.getResult().group.count);
    std::printf("saved detect result to %s\n", output_path.c_str());
    return 0;
}

int run_bench_mode(
    AwnnEngine &mapped_engine,
    AwnnEngine &host_copy_engine,
    YoloV5DetectPostProcessAwnn &processor,
    const std::vector<uint8_t> &input_nchw,
    const int model_h,
    const int model_w,
    const int loops)
{
    if (!yolov5_detect_awnn_bench::verify_output_paths_match(
            mapped_engine, host_copy_engine, processor, input_nchw, model_h, model_w))
    {
        return 1;
    }

    std::printf("\noutput bytes per head:");
    for (std::uint32_t i = 0; i < host_copy_engine.output_count(); ++i)
    {
        std::printf(" [%u]=%u", i, host_copy_engine.output_buffer_byte_size(i));
    }
    std::printf("\n");

    const yolov5_detect_awnn_bench::BenchStats mapped_stats =
        yolov5_detect_awnn_bench::bench_output_path(
            mapped_engine, processor, input_nchw, model_h, model_w, kBenchWarmup, loops);
    const yolov5_detect_awnn_bench::BenchStats copy_stats =
        yolov5_detect_awnn_bench::bench_output_path(
            host_copy_engine, processor, input_nchw, model_h, model_w, kBenchWarmup, loops);

    yolov5_detect_awnn_bench::print_bench_compare(mapped_stats, copy_stats, kBenchWarmup, loops);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    const fs::path base = fs::current_path();
    const std::string default_model = app_resource("yolov5.nb").string();
    const std::string default_input = app_resource("dog.jpg").string();
    const std::string default_output = app_resource("yolov5_detect_awnn_out.jpg").string();

    bool bench_mode = false;
    int bench_loops = kBenchDefaultLoops;
    int argi = 1;

    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0)
    {
        bench_mode = true;
        argi = 2;
        if (argi < argc && argv[argi][0] >= '0' && argv[argi][0] <= '9')
        {
            bench_loops = std::atoi(argv[argi]);
            ++argi;
        }
    }

    const char *model_arg = (argi < argc) ? argv[argi] : default_model.c_str();
    const char *input_arg = (argi + 1 < argc) ? argv[argi + 1] : default_input.c_str();
    const char *output_arg = (argi + 2 < argc) ? argv[argi + 2] : default_output.c_str();

    const std::string model_path = resolve_path(model_arg, base);
    const std::string input_path = resolve_path(input_arg, base);
    const std::string output_path = resolve_path(output_arg, base);

    std::printf("yolov5_detect_awnn\n");
    std::printf("  model : %s\n", model_path.c_str());
    std::printf("  input : %s\n", input_path.c_str());
    if (!bench_mode)
    {
        std::printf("  output: %s\n", output_path.c_str());
    }
    else
    {
        std::printf("  mode  : bench (loops=%d, warmup=%d)\n", bench_loops, kBenchWarmup);
    }

    if (!file_exists(model_path))
    {
        std::fprintf(stderr, "model not found: %s\n", model_path.c_str());
        return 1;
    }
    if (!file_exists(input_path))
    {
        std::fprintf(stderr, "input not found: %s\n", input_path.c_str());
        return 1;
    }

    AwnnEngine::Params engine_params;
    engine_params.model_path = model_path;
    engine_params.output_fetch = OutputFetch::HostCopy;

    AwnnEngine engine(engine_params);
    if (!engine.is_valid())
    {
        std::fprintf(
            stderr,
            "AwnnEngine init failed (check lib/ has libNBGlinker.so + libVIPhal.so, "
            "or set LD_LIBRARY_PATH=$PWD/lib)\n");
        return 1;
    }

    const int model_w = static_cast<int>(engine.input_width());
    const int model_h = static_cast<int>(engine.input_height());
    const int model_c = static_cast<int>(engine.input_channels());
    std::printf("  model input: %dx%dx%d (C×H×W, from .nb)\n", model_c, model_h, model_w);

    cv::Mat model_input;
    std::vector<uint8_t> input_nchw;
    if (!prepare_input_nchw(engine, input_path, model_input, input_nchw))
    {
        return 1;
    }

    YoloV5DetectPostProcessAwnn::Params post_params;
    YoloV5DetectPostProcessAwnn processor(post_params);

    if (bench_mode)
    {
        AwnnEngine::Params mapped_params = engine_params;
        mapped_params.output_fetch = OutputFetch::Mapped;
        AwnnEngine mapped_engine(mapped_params);
        if (!mapped_engine.is_valid())
        {
            std::fprintf(stderr, "AwnnEngine (Mapped) init failed\n");
            return 1;
        }

        return run_bench_mode(
            mapped_engine, engine, processor, input_nchw, model_h, model_w, bench_loops);
    }

    return run_demo(engine, input_nchw, model_input, output_path, model_h, model_w);
}
