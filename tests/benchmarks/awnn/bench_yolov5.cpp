/**
 * YOLOv5 AWNN 全流程 benchmark（自包含，无共享 bench 框架）
 *
 * 用法:
 *   bench_yolov5 [loops] [model.nb] [input.jpg]
 */
#include "deploy_percept/core/engine/AwnnEngine.hpp"
#include "deploy_percept/core/engine/EngineResultGuard.hpp"
#include "deploy_percept/core/post_process/YoloV5DetectPostProcessAwnn.hpp"
#include "deploy_percept/utils/pre_process/image_pre_process.hpp"
#include "test_common/paths.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

namespace {

using deploy_percept::engine::AwnnEngine;
using deploy_percept::engine::EngineResultGuard;
using deploy_percept::engine::VipLiteRuntime;
using deploy_percept::post_process::YoloV5DetectPostProcessAwnn;
using deploy_percept::pre_process::yolov5_preprocess_awnn;
using percept::test::fixture_data;

constexpr int kWarmup = 5;
constexpr int kDefaultLoops = 50;
constexpr const char *kDefaultModel = "yolov5_a733.nb";
constexpr const char *kDefaultInput = "dog.jpg";
constexpr const char *kPostProcessorClass = "YoloV5DetectPostProcessAwnn";

struct BenchRow
{
    const char *tag;
    bool is_copy_host;
};

/** 测试清单：每行对应报告一行 */
constexpr BenchRow kRows[] = {
    {"mapped", false},
    {"host", true},
};

struct BenchRowStats
{
    BenchRow row{};
    double pre_ms{0};
    double infer_ms{0};
    double post_ms{0};
    double pipeline_ms{0};
};

std::string default_awnn_fixture(const char *relative)
{
    return fixture_data(std::string("integration/awnn/") + relative).string();
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

void parse_model_input_dims_whcn(
    const std::array<std::uint32_t, 6> &sizes,
    int &model_w,
    int &model_h,
    int &model_c)
{
    model_w = static_cast<int>(sizes[0]);
    model_h = static_cast<int>(sizes[1]);
    model_c = static_cast<int>(sizes[2]);
}

bool run_preprocess(
    const cv::Mat &cached_bgr,
    const int model_w,
    const int model_h,
    const int model_c,
    std::vector<std::uint8_t> &input_buffer)
{
    if (cached_bgr.empty())
    {
        return false;
    }
    return yolov5_preprocess_awnn(cached_bgr, model_w, model_h, model_c, input_buffer);
}

BenchRowStats bench_one_row(
    AwnnEngine &engine,
    YoloV5DetectPostProcessAwnn &processor,
    const std::string &input_path,
    const int model_w,
    const int model_h,
    const int model_c,
    const BenchRow &row,
    const int loops)
{
    BenchRowStats stats{};
    stats.row = row;

    const cv::Mat cached_bgr = cv::imread(input_path, cv::IMREAD_COLOR);
    if (cached_bgr.empty())
    {
        std::fprintf(stderr, "failed to read input: %s\n", input_path.c_str());
        return stats;
    }

    const std::size_t input_byte_size =
        static_cast<std::size_t>(model_w) * static_cast<std::size_t>(model_h) *
        static_cast<std::size_t>(model_c);

    std::vector<std::uint8_t> input_buffer(input_byte_size);

    double pre_sum = 0;
    double infer_sum = 0;
    double post_sum = 0;

    const int total = kWarmup + loops;
    for (int i = 0; i < total; ++i)
    {
        const auto pre_t0 = std::chrono::steady_clock::now();
        if (!run_preprocess(cached_bgr, model_w, model_h, model_c, input_buffer))
        {
            std::fprintf(stderr, "preprocess failed for tag=%s\n", row.tag);
            return stats;
        }
        const auto pre_t1 = std::chrono::steady_clock::now();

        const auto infer_t0 = std::chrono::steady_clock::now();
        if (!engine.run(input_buffer.data(), input_buffer.size()))
        {
            std::fprintf(stderr, "engine.run failed for tag=%s\n", row.tag);
            return stats;
        }
        const auto infer_t1 = std::chrono::steady_clock::now();

        const auto post_t0 = std::chrono::steady_clock::now();
        bool post_ok = false;
        if (row.is_copy_host)
        {
            const auto &views = engine.getResult().tensor_views;
            post_ok = !views.empty() && processor.run(views);
            engine.releaseResult();
        }
        else
        {
            EngineResultGuard guard(engine);
            const auto &views = guard.getEngineResult().tensor_views;
            post_ok = !views.empty() && processor.run(views);
        }
        const auto post_t1 = std::chrono::steady_clock::now();

        if (!post_ok)
        {
            std::fprintf(stderr, "post failed for tag=%s\n", row.tag);
            return stats;
        }

        if (i >= kWarmup)
        {
            pre_sum += std::chrono::duration<double, std::milli>(pre_t1 - pre_t0).count();
            infer_sum += std::chrono::duration<double, std::milli>(infer_t1 - infer_t0).count();
            post_sum += std::chrono::duration<double, std::milli>(post_t1 - post_t0).count();
        }
    }

    if (loops > 0)
    {
        stats.pre_ms = pre_sum / loops;
        stats.infer_ms = infer_sum / loops;
        stats.post_ms = post_sum / loops;
        stats.pipeline_ms = stats.pre_ms + stats.infer_ms + stats.post_ms;
    }
    return stats;
}

void print_report(
    const char *model_path,
    const char *input_path,
    const int loops,
    const std::vector<BenchRowStats> &rows)
{
    std::printf("\n=== yolov5_detect_awnn full pipeline ===\n");
    std::printf(
        "model=%s  input=%s  warmup=%d  loops=%d\n",
        model_path,
        input_path,
        kWarmup,
        loops);
    std::printf("pipeline = pre + infer + post (ms/frame)\n\n");
    std::printf(
        "%-14s  %-11s  %-11s  %-11s  %8s  %8s  %8s  %8s\n",
        "tag",
        "preprocess",
        "inference",
        "postprocess",
        "pre(ms)",
        "infer(ms)",
        "post(ms)",
        "pipeline");

    for (const BenchRowStats &s : rows)
    {
        const char *inference = s.row.is_copy_host ? "host" : "mapped";
        const char *postprocess = inference;
        std::printf(
            "%-14s  %-11s  %-11s  %-11s  %8.2f  %8.2f  %8.2f  %8.2f\n",
            s.row.tag,
            "opencv_mem",
            inference,
            postprocess,
            s.pre_ms,
            s.infer_ms,
            s.post_ms,
            s.pipeline_ms);
    }

    const BenchRowStats *fastest = nullptr;
    const BenchRowStats *baseline = nullptr;
    for (const BenchRowStats &s : rows)
    {
        if (!s.row.is_copy_host)
        {
            baseline = &s;
        }
        if (fastest == nullptr || s.pipeline_ms < fastest->pipeline_ms)
        {
            fastest = &s;
        }
    }

    if (fastest != nullptr)
    {
        const double fps = fastest->pipeline_ms > 0.0 ? 1000.0 / fastest->pipeline_ms : 0.0;
        std::printf(
            "\n>>> fastest pipeline: %s (%.2f ms, ~%.1f fps)\n",
            fastest->row.tag,
            fastest->pipeline_ms,
            fps);
        if (baseline != nullptr && baseline != fastest && baseline->pipeline_ms > 0.0)
        {
            const double delta =
                (baseline->pipeline_ms - fastest->pipeline_ms) / baseline->pipeline_ms * 100.0;
            std::printf(
                ">>> vs %s (mapped baseline): %+.1f%% pipeline (%.2f ms -> %.2f ms)\n",
                baseline->row.tag,
                delta,
                baseline->pipeline_ms,
                fastest->pipeline_ms);
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    const fs::path base = fs::current_path();

    int loops = kDefaultLoops;
    int argi = 1;
    if (argi < argc && argv[argi][0] >= '0' && argv[argi][0] <= '9')
    {
        loops = std::atoi(argv[argi]);
        ++argi;
    }

    const std::string default_model = default_awnn_fixture(kDefaultModel);
    const std::string default_input = default_awnn_fixture(kDefaultInput);
    const char *model_arg = (argi < argc) ? argv[argi] : default_model.c_str();
    const char *input_arg = (argi + 1 < argc) ? argv[argi + 1] : default_input.c_str();

    const std::string model_path = resolve_path(model_arg, base);
    const std::string input_path = resolve_path(input_arg, base);

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

    static VipLiteRuntime runtime;
    if (!runtime.ok())
    {
        std::fprintf(stderr, "VipLiteRuntime init failed (check LD_LIBRARY_PATH=$PWD/lib)\n");
        return 1;
    }

    std::printf("bench_yolov5\n");
    std::printf("  model  : %s\n", model_path.c_str());
    std::printf("  input  : %s\n", input_path.c_str());
    std::printf("  warmup : %d  loops: %d\n", kWarmup, loops);
    std::printf("  post   : %s\n", kPostProcessorClass);
    std::printf("\n  test plan:\n");
    for (const BenchRow &row : kRows)
    {
        std::printf(
            "  [%s] image_pre_process(opencv_mem)  AwnnEngine(is_copy_host=%s)  %s(tensor=%s)\n",
            row.tag,
            row.is_copy_host ? "true" : "false",
            kPostProcessorClass,
            row.is_copy_host ? "host" : "mapped");
    }
    std::printf("\n");

    std::vector<BenchRowStats> results;
    results.reserve(std::size(kRows));

    YoloV5DetectPostProcessAwnn processor({});
    bool processor_ready = false;

    for (const BenchRow &row : kRows)
    {
        AwnnEngine::Params params;
        params.model_path = model_path;
        params.is_copy_host = row.is_copy_host;

        AwnnEngine engine(params);
        if (!engine.is_valid())
        {
            std::fprintf(stderr, "AwnnEngine init failed for tag=%s\n", row.tag);
            continue;
        }

        int model_w = 0;
        int model_h = 0;
        int model_c = 0;
        parse_model_input_dims_whcn(engine.getInfo().input_sizes.at(0), model_w, model_h, model_c);

        if (!processor_ready)
        {
            YoloV5DetectPostProcessAwnn::Params post_params;
            post_params.model_in_w = model_w;
            post_params.model_in_h = model_h;
            processor = YoloV5DetectPostProcessAwnn(post_params);
            processor_ready = true;
            std::printf(
                "  model input VIP sizes: W=%d H=%d C=%d\n",
                model_w,
                model_h,
                model_c);
        }

        std::printf("\n--- running [%s] ---\n", row.tag);
        results.push_back(
            bench_one_row(engine, processor, input_path, model_w, model_h, model_c, row, loops));
    }

    print_report(model_path.c_str(), input_path.c_str(), loops, results);
    return 0;
}
