/**
 * Allwinner YOLOv5 检测 demo（AwnnEngine + FP32/INT8 后处理）
 *
 * 默认资源：apps/yolov5_detect_awnn/ → share/percept/apps/yolov5_detect_awnn/
 * 用法：yolov5_detect_awnn [model.nb] [input.jpg] [output.jpg]
 */
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcess.hpp"
#include "yolov5_fp32_post.hpp"

namespace fs = std::filesystem;
using deploy_percept::engine::AwnnEngine;
using deploy_percept::post_process::YoloV5DetectPostProcess;

namespace
{

constexpr int kModelW = 640;
constexpr int kModelH = 640;

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

void log_loaded_viplite_libs()
{
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line))
    {
        if (line.find("VIPhal") != std::string::npos ||
            line.find("NBGlinker") != std::string::npos)
        {
            std::printf("  %s\n", line.c_str());
        }
    }
}

/** BGR 640x640 → NCHW RGB uint8（yolov5.nb 输入 layout） */
std::vector<uint8_t> pack_nchw_rgb_uint8(const cv::Mat &bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::vector<uint8_t> nchw(static_cast<std::size_t>(kModelW * kModelH * 3));
    for (int h = 0; h < kModelH; ++h)
    {
        for (int w = 0; w < kModelW; ++w)
        {
            const cv::Vec3b pix = rgb.at<cv::Vec3b>(h, w);
            for (int c = 0; c < 3; ++c)
            {
                nchw[c * kModelH * kModelW + h * kModelW + w] = pix[c];
            }
        }
    }
    return nchw;
}

} // namespace

int main(int argc, char **argv)
{
    const fs::path base = fs::current_path();
    const std::string default_model = app_resource("yolov5.nb").string();
    const std::string default_input = app_resource("dog.jpg").string();
    const std::string default_output = app_resource("yolov5_detect_awnn_out.jpg").string();

    const char *model_arg = (argc > 1) ? argv[1] : default_model.c_str();
    const char *input_arg = (argc > 2) ? argv[2] : default_input.c_str();
    const char *output_arg = (argc > 3) ? argv[3] : default_output.c_str();

    const std::string model_path = resolve_path(model_arg, base);
    const std::string input_path = resolve_path(input_arg, base);
    const std::string output_path = resolve_path(output_arg, base);

    std::printf("yolov5_detect_awnn\n");
    std::printf("  model : %s\n", model_path.c_str());
    std::printf("  input : %s\n", input_path.c_str());
    std::printf("  output: %s\n", output_path.c_str());

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

    std::printf("loaded VIPLite libraries:\n");
    log_loaded_viplite_libs();

    AwnnEngine engine(engine_params);
    if (!engine.is_valid())
    {
        std::fprintf(
            stderr,
            "AwnnEngine init failed (check lib/ has libNBGlinker.so + libVIPhal.so, "
            "or set LD_LIBRARY_PATH=$PWD/lib)\n");
        return 1;
    }

    cv::Mat orig = cv::imread(input_path, cv::IMREAD_COLOR);
    if (orig.empty())
    {
        std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
        return 1;
    }
    std::fprintf(stderr, "loaded input image: %dx%d\n", orig.cols, orig.rows);

    cv::Mat model_input;
    cv::resize(orig, model_input, cv::Size(kModelW, kModelH));

    std::vector<uint8_t> input_nchw = pack_nchw_rgb_uint8(model_input);
    std::fprintf(
        stderr,
        "packed input: %zu bytes (model expects %u)\n",
        input_nchw.size(),
        engine.input_buffer_byte_size());

    if (input_nchw.size() < engine.input_buffer_byte_size())
    {
        std::fprintf(stderr, "input size mismatch with model\n");
        return 1;
    }

    if (!engine.run(input_nchw.data(), input_nchw.size()))
    {
        std::fprintf(stderr, "AwnnEngine::run failed\n");
        return 1;
    }

    cv::Mat result_img = model_input.clone();

    if (engine.outputs_are_fp32())
    {
        float **raw_outputs = engine.output_buffers_float();
        if (raw_outputs == nullptr || engine.output_count() != 3)
        {
            std::fprintf(stderr, "AwnnEngine FP32 output buffers unavailable\n");
            return 1;
        }

        float *outputs[3] = {raw_outputs[0], raw_outputs[1], raw_outputs[2]};
        std::vector<yolov5_fp32_post::Detection> detections;
        std::fprintf(stderr, "post process (fp32)...\n");
        const int det_count = yolov5_fp32_post::run(outputs, kModelW, kModelH, detections);
        yolov5_fp32_post::draw(result_img, detections);

        std::error_code ec;
        fs::create_directories(fs::path(output_path).parent_path(), ec);
        if (!cv::imwrite(output_path, result_img))
        {
            std::fprintf(stderr, "failed to save output to %s\n", output_path.c_str());
            return 1;
        }

        std::printf("detection num: %d\n", det_count);
        std::printf("saved detect result to %s\n", output_path.c_str());
        return 0;
    }

    if (!engine.outputs_are_int8())
    {
        std::fprintf(stderr, "unsupported model output format\n");
        return 1;
    }

    int8_t **raw_outputs = engine.output_buffers();
    if (raw_outputs == nullptr || engine.output_count() != 3)
    {
        std::fprintf(stderr, "AwnnEngine output buffers unavailable\n");
        return 1;
    }

    // NPU raw INT8 → 后处理通用输入（与 yolov5_detect_rknn 一致）
    std::vector<int8_t *> output_buffers_int8;
    std::vector<int32_t> output_zps;
    std::vector<float> output_scales;
    output_buffers_int8.reserve(engine.output_count());
    output_zps.reserve(engine.output_count());
    output_scales.reserve(engine.output_count());
    for (std::uint32_t i = 0; i < engine.output_count(); ++i)
    {
        output_buffers_int8.push_back(raw_outputs[i]);
        output_zps.push_back(engine.output_zero_point(i));
        output_scales.push_back(engine.output_scale(i));
    }

    YoloV5DetectPostProcess::Params post_params;
    YoloV5DetectPostProcess processor(post_params);

    std::fprintf(stderr, "post process (int8)...\n");
    if (!processor.run(output_buffers_int8, kModelH, kModelW, output_zps, output_scales))
    {
        std::fprintf(stderr, "post process failed: %s\n", processor.getResult().message.c_str());
        return 1;
    }

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
