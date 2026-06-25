/**
 * Allwinner YOLOv5 检测 demo（AwnnEngine + deploy_percept 后处理）
 *
 * 默认资源：apps/yolov5_detect_awnn/ → share/percept/apps/yolov5_detect_awnn/
 * 用法：yolov5_detect_awnn [model.nb] [input.jpg] [output.jpg]
 */
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/post_process/YoloV5DetectPostProcessAwnn.hpp"

namespace fs = std::filesystem;
using deploy_percept::engine::AwnnEngine;
using deploy_percept::post_process::YoloV5DetectPostProcessAwnn;

namespace
{

constexpr int kModelW = 640;
constexpr int kModelH = 640;

fs::path app_resource(const char *relative)
{
    if (const char *root = std::getenv("PERCEPT_ROOT"))
    {
        return fs::path(root) / "apps" / "yolov5_detect_awnn" / relative;
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

/** BGR 640x640 → NCHW RGB uint8（yolov5.nb 输入 layout） */
std::vector<uint8_t> pack_nchw_rgb_uint8(const cv::Mat &bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::vector<uint8_t> nchw(static_cast<size_t>(kModelW * kModelH * 3));
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

    cv::Mat orig = cv::imread(input_path, cv::IMREAD_COLOR);
    if (orig.empty())
    {
        std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
        return 1;
    }

    cv::Mat model_input;
    cv::resize(orig, model_input, cv::Size(kModelW, kModelH));

    AwnnEngine::Params engine_params;
    engine_params.model_path = model_path;
    AwnnEngine engine(engine_params);
    if (!engine.is_valid())
    {
        return 1;
    }

    std::vector<uint8_t> input_nchw = pack_nchw_rgb_uint8(model_input);
    if (!engine.run(input_nchw.data()))
    {
        std::fprintf(stderr, "AwnnEngine::run failed\n");
        return 1;
    }

    float **outputs = engine.output_buffers();
    if (outputs == nullptr)
    {
        std::fprintf(stderr, "AwnnEngine output buffers unavailable\n");
        return 1;
    }

    YoloV5DetectPostProcessAwnn::Params post_params;
    post_params.conf_threshold = 0.4f;
    YoloV5DetectPostProcessAwnn processor(post_params);

    if (!processor.run({outputs[0], outputs[1], outputs[2]}, kModelH, kModelW))
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
