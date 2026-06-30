/**
 * Allwinner YOLOv8 检测 demo（AwnnEngine + YoloV8DetectPostProcessAwnn）
 *
 * 用法（路径相对当前工作目录，或使用绝对路径）：
 *   yolov8_detect_awnn [model.nb] [input.jpg] [output.jpg]
 *
 * 默认：./yolov8.nb  ./dog.jpg  ./yolov8_detect_awnn_out.jpg
 */
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/engine/AwnnImageInput.hpp"
#include "deploy_percept/engine/VipLiteRuntime.hpp"
#include "deploy_percept/engine/AwnnResultGuard.hpp"
#include "deploy_percept/post_process/YoloV8DetectPostProcessAwnn.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::AwnnResultGuard;
using deploy_percept::engine::AwnnRgbInputShape;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::YoloV8DetectPostProcessAwnn;

namespace
{

constexpr const char *kDefaultModel = "yolov8.nb";
constexpr const char *kDefaultInput = "dog.jpg";
constexpr const char *kDefaultOutput = "yolov8_detect_awnn_out.jpg";

} // namespace

int main(int argc, char **argv)
{
    const char *model_path = (argc > 1) ? argv[1] : kDefaultModel;
    const char *input_path = (argc > 2) ? argv[2] : kDefaultInput;
    const char *output_path = (argc > 3) ? argv[3] : kDefaultOutput;

    std::printf("yolov8_detect_awnn\n");
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

    const AwnnRgbInputShape input_shape = deploy_percept::engine::resolveRgbInputShape(engine.getInfo());
    std::printf("  model input: W=%d H=%d C=%d bytes=%zu\n",
        input_shape.width, input_shape.height, input_shape.channels, input_shape.buffer_bytes);

    cv::Mat orig_bgr;
    std::vector<std::uint8_t> input_buffer;
    if (!deploy_percept::engine::prepareLetterboxRgbInput(
            input_path, input_shape, orig_bgr, input_buffer))
    {
        std::fprintf(stderr, "failed to prepare input: %s\n", input_path);
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

    YoloV8DetectPostProcessAwnn::Params post_params;
    post_params.model_in_w = input_shape.width;
    post_params.model_in_h = input_shape.height;
    post_params.orig_img_w = orig_bgr.cols;
    post_params.orig_img_h = orig_bgr.rows;

    YoloV8DetectPostProcessAwnn processor(post_params);
    if (!processor.run(engine_result_guard.views()))
    {
        std::fprintf(stderr, "post process failed: %s\n", processor.getResult().message.c_str());
        return 1;
    }

    cv::Mat result_img = orig_bgr.clone();
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
