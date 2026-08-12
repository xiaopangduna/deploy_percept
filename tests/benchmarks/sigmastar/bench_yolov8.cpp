/**
 * YOLOv8 Detect Sigmastar 全流程 benchmark（自包含，无共享 bench 框架）
 *
 * 用法:
 *   bench_yolov8 [loops] [model.img] [input.jpg]
 */
#include "deploy_percept/core/engine/EngineResultGuard.hpp"
#include "deploy_percept/core/engine/SigmastarEngine.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessSigmastar.hpp"
#include "test_common/paths.hpp"

#include "mi_ipu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace {

using deploy_percept::engine::EngineResultGuard;
using deploy_percept::engine::SigmastarEngine;
using deploy_percept::post_process::YoloV8DetectPostProcessSigmastar;
using percept::test::fixture_data;

constexpr int kWarmup = 5;
constexpr int kDefaultLoops = 50;
constexpr const char *kDefaultModel = "yolov8n_640x640_ssd2355.img";
constexpr const char *kDefaultInput = "dog.jpg";
constexpr const char *kPostProcessorClass = "YoloV8DetectPostProcessSigmastar";
constexpr std::uint32_t kDivpYuvAlignment = 16;

struct BenchRow
{
    const char *tag;
    bool is_copy_host;
};

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

std::string default_sigmastar_fixture(const char *relative)
{
    return fixture_data(std::string("integration/sigmastar/") + relative).string();
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

std::uint32_t align_up(std::uint32_t x, std::uint32_t align)
{
    return (x + align - 1u) & ~(align - 1u);
}

std::uint32_t align_down(std::uint32_t x, std::uint32_t align)
{
    return (x / align) * align;
}

void yv12_to_nv12(const cv::Mat &yv12, cv::Mat &nv12)
{
    const int width = yv12.cols;
    const int height = yv12.rows * 2 / 3;
    const std::size_t stride = static_cast<std::size_t>(yv12.step[0]);
    yv12.copyTo(nv12);

    cv::Mat in_v(
        height / 2,
        width / 2,
        CV_8UC1,
        yv12.data + stride * static_cast<std::size_t>(height),
        stride / 2);
    cv::Mat in_u(
        height / 2,
        width / 2,
        CV_8UC1,
        yv12.data + stride * static_cast<std::size_t>(height) +
            (stride / 2) * static_cast<std::size_t>(height / 2),
        stride / 2);

    for (int row = 0; row < height / 2; ++row)
    {
        for (int col = 0; col < width / 2; ++col)
        {
            nv12.at<uchar>(height + row, 2 * col) = in_u.at<uchar>(row, col);
            nv12.at<uchar>(height + row, 2 * col + 1) = in_v.at<uchar>(row, col);
        }
    }
}

bool pack_strided_rows(
    const std::uint8_t *src,
    int src_row_bytes,
    int copy_bytes,
    int num_rows,
    std::uint32_t dst_stride,
    std::size_t buffer_bytes,
    std::vector<std::uint8_t> &input_buffer)
{
    if (static_cast<std::size_t>(num_rows) * dst_stride > buffer_bytes)
    {
        return false;
    }
    auto *dst = input_buffer.data();
    for (int row = 0; row < num_rows; ++row)
    {
        std::memcpy(dst, src, static_cast<std::size_t>(copy_bytes));
        dst += dst_stride;
        src += src_row_bytes;
    }
    return true;
}

bool run_preprocess(
    const cv::Mat &cached_bgr,
    const int model_w,
    const int model_h,
    const std::uint32_t elm_format,
    const std::size_t buffer_bytes,
    std::vector<std::uint8_t> &input_buffer)
{
    if (cached_bgr.empty())
    {
        return false;
    }

    cv::Mat model_input_bgr;
    cv::resize(cached_bgr, model_input_bgr, cv::Size(model_w, model_h));
    input_buffer.assign(buffer_bytes, 0);

    auto fmt = static_cast<MI_IPU_ELEMENT_FORMAT>(elm_format);
    if (fmt == MI_IPU_FORMAT_NV12)
    {
        const int align_h = static_cast<int>(align_down(static_cast<std::uint32_t>(model_h), 2));
        const int align_w = static_cast<int>(align_down(static_cast<std::uint32_t>(model_w), 2));
        cv::Mat align_bgr = model_input_bgr(cv::Range(0, align_h), cv::Range(0, align_w)).clone();
        cv::Mat yv12;
        cv::cvtColor(align_bgr, yv12, cv::COLOR_BGR2YUV_YV12);
        cv::Mat nv12;
        yv12_to_nv12(yv12, nv12);

        const std::uint32_t row_stride =
            align_up(static_cast<std::uint32_t>(model_w), kDivpYuvAlignment);
        return pack_strided_rows(
            nv12.data,
            model_w,
            model_w,
            nv12.rows,
            row_stride,
            buffer_bytes,
            input_buffer);
    }

    if (fmt == MI_IPU_FORMAT_ARGB8888)
    {
        cv::Mat bgra;
        cv::cvtColor(model_input_bgr, bgra, cv::COLOR_BGR2BGRA);
        const std::uint32_t row_stride =
            align_up(static_cast<std::uint32_t>(model_w) * 4u, kDivpYuvAlignment);
        return pack_strided_rows(
            bgra.ptr<std::uint8_t>(0),
            static_cast<int>(bgra.step[0]),
            model_w * 4,
            model_h,
            row_stride,
            buffer_bytes,
            input_buffer);
    }

    if (fmt == MI_IPU_FORMAT_ABGR8888)
    {
        cv::Mat rgba;
        cv::cvtColor(model_input_bgr, rgba, cv::COLOR_BGR2RGBA);
        const std::uint32_t row_stride =
            align_up(static_cast<std::uint32_t>(model_w) * 4u, kDivpYuvAlignment);
        return pack_strided_rows(
            rgba.ptr<std::uint8_t>(0),
            static_cast<int>(rgba.step[0]),
            model_w * 4,
            model_h,
            row_stride,
            buffer_bytes,
            input_buffer);
    }

    return false;
}

BenchRowStats bench_one_row(
    SigmastarEngine &engine,
    YoloV8DetectPostProcessSigmastar &processor,
    const std::string &input_path,
    const int model_w,
    const int model_h,
    const std::uint32_t elm_format,
    const std::size_t buffer_bytes,
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

    std::vector<std::uint8_t> input_buffer(buffer_bytes);

    double pre_sum = 0;
    double infer_sum = 0;
    double post_sum = 0;

    const int total = kWarmup + loops;
    for (int i = 0; i < total; ++i)
    {
        const auto pre_t0 = std::chrono::steady_clock::now();
        if (!run_preprocess(cached_bgr, model_w, model_h, elm_format, buffer_bytes, input_buffer))
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
    std::printf("\n=== yolov8_detect_sigmastar full pipeline ===\n");
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

    const std::string default_model = default_sigmastar_fixture(kDefaultModel);
    const std::string default_input = default_sigmastar_fixture(kDefaultInput);
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

    std::printf("bench_yolov8\n");
    std::printf("  model  : %s\n", model_path.c_str());
    std::printf("  input  : %s\n", input_path.c_str());
    std::printf("  warmup : %d  loops: %d\n", kWarmup, loops);
    std::printf("  post   : %s\n", kPostProcessorClass);
    std::printf("\n  test plan:\n");
    for (const BenchRow &row : kRows)
    {
        std::printf(
            "  [%s] opencv preprocess  SigmastarEngine(is_copy_host=%s)  %s(tensor=%s)\n",
            row.tag,
            row.is_copy_host ? "true" : "false",
            kPostProcessorClass,
            row.is_copy_host ? "host" : "mapped");
    }
    std::printf("\n");

    std::vector<BenchRowStats> results;
    results.reserve(std::size(kRows));

    YoloV8DetectPostProcessSigmastar processor({});
    bool processor_ready = false;
    int model_w = 0;
    int model_h = 0;
    std::uint32_t elm_format = 0;
    std::size_t buffer_bytes = 0;

    for (const BenchRow &row : kRows)
    {
        SigmastarEngine::Params params;
        params.model_path = model_path;
        params.is_copy_host = row.is_copy_host;

        SigmastarEngine engine(params);
        if (!engine.is_valid())
        {
            std::fprintf(stderr, "SigmastarEngine init failed for tag=%s\n", row.tag);
            continue;
        }

        const auto &info = engine.getInfo();
        if (info.input_sizes.empty() || info.input_byte_sizes.empty() ||
            info.input_elm_formats.empty())
        {
            std::fprintf(stderr, "SigmastarEngine has no input tensors for tag=%s\n", row.tag);
            continue;
        }

        model_h = static_cast<int>(info.input_sizes[0][1]);
        model_w = static_cast<int>(info.input_sizes[0][2]);
        elm_format = info.input_elm_formats[0];
        buffer_bytes = info.input_byte_sizes[0];

        if (!processor_ready)
        {
            YoloV8DetectPostProcessSigmastar::Params post_params;
            post_params.model_in_w = model_w;
            post_params.model_in_h = model_h;
            post_params.output_shapes = info.output_sizes;
            post_params.output_scalars = info.output_scalars;
            processor = YoloV8DetectPostProcessSigmastar(post_params);
            processor_ready = true;
            std::printf(
                "  model input: H=%d W=%d elm_format=%u bytes=%zu\n",
                model_h,
                model_w,
                elm_format,
                buffer_bytes);
        }

        std::printf("\n--- running [%s] ---\n", row.tag);
        results.push_back(bench_one_row(
            engine,
            processor,
            input_path,
            model_w,
            model_h,
            elm_format,
            buffer_bytes,
            row,
            loops));
    }

    print_report(model_path.c_str(), input_path.c_str(), loops, results);
    return 0;
}
