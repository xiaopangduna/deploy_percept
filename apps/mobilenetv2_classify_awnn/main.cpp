/**
 * Allwinner MobileNetV2 ImageNet 分类 demo（AwnnEngine + ClassificationPostProcessAwnn）
 *
 * 用法（路径相对当前工作目录，或使用绝对路径）：
 *   mobilenetv2_classify_awnn [--duration-sec T] [model.nb] [input.jpg] [model_outputs.npz]
 *
 * 默认：./mobilenetv2_a733.nb  ./cat.jpg
 * --duration-sec：按墙钟循环推理，供 tests/profiling/percept_monitor 旁路采样
 * 延迟对比见 tests/benchmarks/awnn/bench_mobilenetv2_classify
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_percept/core/engine/AwnnEngine.hpp"
#include "deploy_percept/core/engine/EngineResultGuard.hpp"
#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"
#include "deploy_percept/utils/npy.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::EngineResultGuard;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::ClassificationPostProcessAwnn;
using deploy_percept::utils::convertTensorViewsToNpz;
using deploy_percept::utils::save_npz;

namespace
{

constexpr const char *kDefaultModel = "mobilenetv2_a733.nb";
constexpr const char *kDefaultInput = "cat.jpg";

bool prepare_model_input(
    const std::string &input_path,
    const int model_w,
    const int model_h,
    const std::size_t buffer_bytes,
    std::vector<std::uint8_t> &input_buffer)
{
    cv::Mat orig = cv::imread(input_path, cv::IMREAD_COLOR);
    if (orig.empty())
    {
        std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
        return false;
    }

    cv::Mat resized;
    cv::resize(orig, resized, cv::Size(model_w, model_h));
    input_buffer.assign(buffer_bytes, 0);

    cv::Mat rgb_hwc(model_h, model_w, CV_8UC3, input_buffer.data());
    cv::cvtColor(resized, rgb_hwc, cv::COLOR_BGR2RGB);
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    double duration_sec = 0.0;
    int argi = 1;
    if (argi + 1 < argc && std::strcmp(argv[argi], "--duration-sec") == 0)
    {
        duration_sec = std::atof(argv[argi + 1]);
        if (duration_sec <= 0.0)
        {
            std::fprintf(stderr, "invalid --duration-sec: %s\n", argv[argi + 1]);
            return 1;
        }
        argi += 2;
    }

    const std::string model_path_str =
        (argi < argc) ? argv[argi] : kDefaultModel;
    const std::string input_path_str =
        (argi + 1 < argc) ? argv[argi + 1] : kDefaultInput;
    const char *model_path = model_path_str.c_str();
    const char *input_path = input_path_str.c_str();
    const char *npz_path = (argi + 2 < argc) ? argv[argi + 2] : nullptr;

    std::printf("mobilenetv2_classify_awnn\n");
    std::printf("  model : %s\n", model_path);
    std::printf("  input : %s\n", input_path);
    if (duration_sec > 0.0)
    {
        std::printf("  duration-sec: %.3f\n", duration_sec);
    }
    if (npz_path != nullptr)
    {
        std::printf("  npz   : %s\n", npz_path);
    }

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

    AwnnEngine::Params engine_params;
    engine_params.model_path = model_path;

    AwnnEngine engine(engine_params);
    if (!engine.ok())
    {
        std::fprintf(stderr, "AwnnEngine init failed\n");
        return 1;
    }

    std::printf("%s", engine.dump_info().c_str());

    // VIP input_sizes 顺序 [W,H,C,N]；demo 当前按 sizes[1]=H, sizes[2]=W 解析
    const auto &sizes = engine.getInfo().input_sizes.at(0);
    const int model_h = static_cast<int>(sizes[1]);
    const int model_w = static_cast<int>(sizes[2]);
    const std::size_t buffer_bytes = engine.getInfo().input_byte_sizes.at(0);

    std::vector<std::uint8_t> input_buffer;
    if (!prepare_model_input(input_path, model_w, model_h, buffer_bytes, input_buffer))
    {
        return 1;
    }

    ClassificationPostProcessAwnn processor({
        .num_classes = static_cast<int>(engine.getInfo().output_sizes.at(0)[0]),
    });

    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t frames = 0;
    do
    {
        if (!engine.run(input_buffer.data(), input_buffer.size()))
        {
            std::fprintf(stderr, "AwnnEngine::run failed\n");
            return 1;
        }

        EngineResultGuard engine_result_guard(engine);
        const auto &tensor_views = engine_result_guard.getEngineResult().tensor_views;
        if (tensor_views.empty())
        {
            std::fprintf(stderr, "AwnnEngine output tensors unavailable\n");
            return 1;
        }

        if (npz_path != nullptr && frames == 0)
        {
            std::error_code ec;
            try
            {
                const cnpy::npz_t model_outputs_npz = convertTensorViewsToNpz(tensor_views);
                fs::create_directories(fs::path(npz_path).parent_path(), ec);
                if (!save_npz(npz_path, model_outputs_npz))
                {
                    std::fprintf(stderr, "failed to save model outputs to %s\n", npz_path);
                    return 1;
                }
                std::printf("saved model outputs to %s\n", npz_path);
            }
            catch (const std::exception &ex)
            {
                std::fprintf(stderr, "failed to dump npz: %s\n", ex.what());
                return 1;
            }
        }

        if (!processor.run(tensor_views))
        {
            std::fprintf(stderr, "post process failed\n");
            return 1;
        }
        ++frames;
    } while (duration_sec > 0.0 &&
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() <
                 duration_sec);

    const auto &result = processor.getResult();
    std::printf("top1: class_id=%d score=%.4f\n", result.class_id, result.score);
    if (duration_sec > 0.0)
    {
        const double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("done: frames=%llu wall=%.3fs\n",
                    static_cast<unsigned long long>(frames),
                    wall);
    }

    return 0;
}
