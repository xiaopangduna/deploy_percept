/**
 * Allwinner YOLOv5 检测 demo（AwnnEngine + YoloV5DetectPostProcessAwnn）
 *
 * 用法（路径相对当前工作目录，或使用绝对路径）：
 *   yolov5_detect_awnn [model.nb] [input.jpg] [output.jpg]
 *
 * 默认：./yolov5.nb  ./dog.jpg  ./yolov5_detect_awnn_out.jpg
 * 性能对比见 benchmarks/awnn/bench_yolov5_detect_awnn_mapped_vs_hostcopy
 */
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::OutputFetch;
using deploy_percept::post_process::YoloV5DetectPostProcessAwnn;

namespace
{

constexpr const char *kDefaultModel = "yolov5.nb";
constexpr const char *kDefaultInput = "dog.jpg";
constexpr const char *kDefaultOutput = "yolov5_detect_awnn_out.jpg";

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
    std::vector<std::uint8_t> &input_nchw)
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

} // namespace

int main(int argc, char **argv)
{
    const char *model_path = (argc > 1) ? argv[1] : kDefaultModel;
    const char *input_path = (argc > 2) ? argv[2] : kDefaultInput;
    const char *output_path = (argc > 3) ? argv[3] : kDefaultOutput;

    std::printf("yolov5_detect_awnn\n");
    std::printf("  model : %s\n", model_path);
    std::printf("  input : %s\n", input_path);
    std::printf("  output: %s\n", output_path);

    if (!fs::is_regular_file(model_path))
    {
        std::fprintf(stderr, "model not found: %s\n", model_path);
        return 1;
    }
    if (!fs::is_regular_file(input_path))
    {
        std::fprintf(stderr, "input not found: %s\n", input_path);
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
            "AwnnEngine init failed (set LD_LIBRARY_PATH to VIPLite lib dir)\n");
        return 1;
    }

    const int model_w = static_cast<int>(engine.input_width());
    const int model_h = static_cast<int>(engine.input_height());
    const int model_c = static_cast<int>(engine.input_channels());
    std::printf("  model input: %dx%dx%d (C×H×W)\n", model_c, model_h, model_w);

    cv::Mat model_input;
    std::vector<std::uint8_t> input_nchw;
    if (!prepare_input_nchw(engine, input_path, model_input, input_nchw))
    {
        return 1;
    }

    if (!engine.run(input_nchw.data(), input_nchw.size()))
    {
        std::fprintf(stderr, "AwnnEngine::run failed\n");
        return 1;
    }

    const std::vector<float *> outputs = collect_output_ptrs(engine);
    if (outputs.empty())
    {
        std::fprintf(stderr, "AwnnEngine output buffers unavailable\n");
        return 1;
    }

    YoloV5DetectPostProcessAwnn processor(YoloV5DetectPostProcessAwnn::Params{});
    if (!processor.run(outputs, model_h, model_w))
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
        std::fprintf(stderr, "failed to save output to %s\n", output_path);
        return 1;
    }

    std::printf("detection num: %d\n", processor.getResult().group.count);
    std::printf("saved detect result to %s\n", output_path);
    return 0;
}
