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
#include "deploy_percept/engine/VipLiteRuntime.hpp"
#include "deploy_percept/engine/AwnnResultGuard.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::AwnnResultGuard;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::YoloV5DetectPostProcessAwnn;

namespace
{

constexpr const char *kDefaultModel = "yolov5.nb";
constexpr const char *kDefaultInput = "dog.jpg";
constexpr const char *kDefaultOutput = "yolov5_detect_awnn_out.jpg";

/**
 * 将 BGR 图像 pack 为 VIP input buffer 字节序。
 * VIP input_sizes 顺序为 [W, H, C, N]（vip_lite.h）；buffer 线性布局 w 最快变。
 * batch=1 时与 ai-sdk yolov5 demo 的 pack 方式一致。
 */
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

bool prepare_model_input(
    const std::string &input_path,
    const int model_w,
    const int model_h,
    const int model_c,
    cv::Mat &model_input,
    std::vector<std::uint8_t> &input_buffer)
{
    cv::Mat orig = cv::imread(input_path, cv::IMREAD_COLOR);
    if (orig.empty())
    {
        std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
        return false;
    }

    cv::resize(orig, model_input, cv::Size(model_w, model_h));
    input_buffer = pack_rgb_for_vip_input(model_input, model_w, model_h, model_c);
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

    VipLiteRuntime runtime;
    if (!runtime.ok())
    {
        std::fprintf(
            stderr,
            "VipLiteRuntime init failed (set LD_LIBRARY_PATH to VIPLite lib dir)\n");
        return 1;
    }

    AwnnEngine::Param engine_params;
    engine_params.model_path = model_path;

    AwnnEngine engine(engine_params);
    if (!engine.is_valid())
    {
        std::fprintf(stderr, "AwnnEngine init failed\n");
        return 1;
    }

    const auto &model_info = engine.getInfo();
    const auto &sizes = model_info.input_sizes.at(0);
    const int model_w = static_cast<int>(sizes[0]);
    const int model_h = static_cast<int>(sizes[1]);
    const int model_c = static_cast<int>(sizes[2]);
    std::printf("  model input VIP sizes: W=%d H=%d C=%d\n", model_w, model_h, model_c);

    cv::Mat model_input;
    std::vector<std::uint8_t> input_buffer;
    if (!prepare_model_input(input_path, model_w, model_h, model_c, model_input, input_buffer))
    {
        return 1;
    }

    if (!engine.run(input_buffer.data(), input_buffer.size()))
    {
        std::fprintf(stderr, "AwnnEngine::run failed\n");
        return 1;
    }

    AwnnResultGuard engine_result_guard(engine);
    if (engine_result_guard.empty())
    {
        std::fprintf(stderr, "AwnnEngine output tensors unavailable\n");
        return 1;
    }

    YoloV5DetectPostProcessAwnn::Params post_params;
    post_params.model_in_w = model_w;
    post_params.model_in_h = model_h;

    YoloV5DetectPostProcessAwnn processor(post_params);
    if (!processor.run(engine_result_guard.views()))
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
