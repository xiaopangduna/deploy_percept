/**
 * Sigmastar YOLOv8 检测批量 dump（低 conf + NMS → JSON）
 *
 * 用法：
 *   yolov8_detect_dump_sigmastar --model <path.img> --image-dir <dir>
 *       [--out dump_board.json] [--conf 0.01] [--nms 0.5] [-h]
 *
 * 板端只写 JSON（meta + images[].detections），不做比对 / mAP。
 * path_img 相对 --image-dir；框为原图像素 xyxy（plain resize 逆变换）。
 */
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "mi_ipu.h"

#include "deploy_percept/core/engine/SigmastarEngine.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessSigmastar.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::SigmastarEngine;
using deploy_percept::engine::SigmastarRuntime;
using deploy_percept::engine::extract_fscalars_from_MI_IPU_TensorDesc_t;
using deploy_percept::engine::extract_shapes_from_MI_IPU_TensorDesc_t;
using deploy_percept::post_process::YoloV8DetectPostProcessSigmastar;
using json = nlohmann::json;

namespace
{

constexpr const char *kDefaultOut = "dump_board.json";
constexpr float kDefaultConf = 0.01f;
constexpr float kDefaultNms = 0.5f;
constexpr int kMaxDet = 300;
constexpr std::uint32_t kDivpYuvAlignment = 16;
constexpr int kFormatVersion = 1;

struct Cli
{
    std::string model;
    std::string image_dir;
    std::string out = kDefaultOut;
    float conf = kDefaultConf;
    float nms = kDefaultNms;
};

std::uint32_t align_up(std::uint32_t x, std::uint32_t align)
{
    return (x + align - 1u) & ~(align - 1u);
}

std::uint32_t align_down(std::uint32_t x, std::uint32_t align)
{
    return (x / align) * align;
}

void print_help(const char *argv0)
{
    std::printf(
        "Usage: %s --model <path.img> --image-dir <dir> [--out %s] "
        "[--conf %.2f] [--nms %.2f] [-h]\n"
        "\n"
        "  --model       Sigmastar .img (required)\n"
        "  --image-dir   image directory, recursive jpg/jpeg/png/bmp (required)\n"
        "  --out         dump JSON path (default: %s)\n"
        "  --conf        dump conf threshold (default: %.2f); PC may filter higher\n"
        "  --nms         NMS IoU threshold (default: %.2f)\n"
        "  -h, --help    show this help\n"
        "\n"
        "Writes JSON: meta + images[].detections (original-pixel xyxy, score desc).\n"
        "max_det fixed at %d (written to meta).\n",
        argv0,
        kDefaultOut,
        static_cast<double>(kDefaultConf),
        static_cast<double>(kDefaultNms),
        kDefaultOut,
        static_cast<double>(kDefaultConf),
        static_cast<double>(kDefaultNms),
        kMaxDet);
}

bool parse_cli(int argc, char **argv, Cli &cli)
{
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0)
        {
            print_help(argv[0]);
            std::exit(0);
        }
        auto need_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (std::strcmp(a, "--model") == 0)
        {
            const char *v = need_value(a);
            if (v == nullptr)
            {
                return false;
            }
            cli.model = v;
        }
        else if (std::strcmp(a, "--image-dir") == 0)
        {
            const char *v = need_value(a);
            if (v == nullptr)
            {
                return false;
            }
            cli.image_dir = v;
        }
        else if (std::strcmp(a, "--out") == 0)
        {
            const char *v = need_value(a);
            if (v == nullptr)
            {
                return false;
            }
            cli.out = v;
        }
        else if (std::strcmp(a, "--conf") == 0)
        {
            const char *v = need_value(a);
            if (v == nullptr)
            {
                return false;
            }
            cli.conf = static_cast<float>(std::atof(v));
            if (!(cli.conf >= 0.f && cli.conf <= 1.f))
            {
                std::fprintf(stderr, "invalid --conf: %s\n", v);
                return false;
            }
        }
        else if (std::strcmp(a, "--nms") == 0)
        {
            const char *v = need_value(a);
            if (v == nullptr)
            {
                return false;
            }
            cli.nms = static_cast<float>(std::atof(v));
            if (!(cli.nms >= 0.f && cli.nms <= 1.f))
            {
                std::fprintf(stderr, "invalid --nms: %s\n", v);
                return false;
            }
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", a);
            return false;
        }
    }
    if (cli.model.empty() || cli.image_dir.empty())
    {
        std::fprintf(stderr, "--model and --image-dir are required\n");
        print_help(argv[0]);
        return false;
    }
    return true;
}

bool is_image_ext(const fs::path &p)
{
    std::string ext = p.extension().string();
    for (char &c : ext)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

/** 相对 image_dir 的 POSIX 路径，按字典序。 */
bool collect_images(const fs::path &image_dir, std::vector<std::string> &rel_paths)
{
    rel_paths.clear();
    std::error_code ec;
    if (!fs::is_directory(image_dir, ec))
    {
        std::fprintf(stderr, "image-dir is not a directory: %s\n", image_dir.c_str());
        return false;
    }
    for (fs::recursive_directory_iterator it(image_dir, ec), end; it != end; it.increment(ec))
    {
        if (ec)
        {
            std::fprintf(stderr, "directory walk error: %s\n", ec.message().c_str());
            return false;
        }
        if (!it->is_regular_file(ec) || ec)
        {
            continue;
        }
        if (!is_image_ext(it->path()))
        {
            continue;
        }
        const fs::path rel = fs::relative(it->path(), image_dir, ec);
        if (ec)
        {
            std::fprintf(stderr, "relative path failed: %s\n", it->path().c_str());
            return false;
        }
        rel_paths.push_back(rel.generic_string());
    }
    std::sort(rel_paths.begin(), rel_paths.end());
    return true;
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
    std::vector<std::uint8_t> &input_buffer,
    const char *tag)
{
    if (static_cast<std::size_t>(num_rows) * dst_stride > buffer_bytes)
    {
        std::fprintf(stderr, "%s buffer too small\n", tag);
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

bool pack_rgb888_tight(
    const cv::Mat &model_input_bgr,
    bool to_rgb,
    const int model_w,
    const int model_h,
    const std::size_t buffer_bytes,
    std::vector<std::uint8_t> &input_buffer,
    const char *tag)
{
    cv::Mat packed;
    if (to_rgb)
    {
        cv::cvtColor(model_input_bgr, packed, cv::COLOR_BGR2RGB);
    }
    else
    {
        packed = model_input_bgr;
    }
    const std::uint32_t row_stride = static_cast<std::uint32_t>(model_w) * 3u;
    return pack_strided_rows(
        packed.ptr<std::uint8_t>(0),
        static_cast<int>(packed.step[0]),
        model_w * 3,
        model_h,
        row_stride,
        buffer_bytes,
        input_buffer,
        tag);
}

const char *elm_format_name(std::uint32_t elm_format)
{
    switch (static_cast<MI_IPU_ELEMENT_FORMAT>(elm_format))
    {
    case MI_IPU_FORMAT_U8:
        return "U8";
    case MI_IPU_FORMAT_NV12:
        return "NV12";
    case MI_IPU_FORMAT_ARGB8888:
        return "ARGB8888(BGRA)";
    case MI_IPU_FORMAT_ABGR8888:
        return "ABGR8888(RGBA)";
    case MI_IPU_FORMAT_GRAY:
        return "GRAY";
#ifdef MI_IPU_FORMAT_RGB888
    case MI_IPU_FORMAT_RGB888:
        return "RGB888";
#endif
#ifdef MI_IPU_FORMAT_BGR888
    case MI_IPU_FORMAT_BGR888:
        return "BGR888";
#endif
    default:
        return "UNKNOWN";
    }
}

bool pack_model_input(
    const cv::Mat &model_input_bgr,
    const int model_w,
    const int model_h,
    const std::uint32_t elm_format,
    const std::size_t buffer_bytes,
    std::vector<std::uint8_t> &input_buffer)
{
    if (model_input_bgr.cols != model_w || model_input_bgr.rows != model_h)
    {
        std::fprintf(
            stderr,
            "resize size mismatch: got %dx%d expect %dx%d\n",
            model_input_bgr.cols,
            model_input_bgr.rows,
            model_w,
            model_h);
        return false;
    }
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
            input_buffer,
            "NV12");
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
            input_buffer,
            "ARGB8888");
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
            input_buffer,
            "ABGR8888");
    }

#ifdef MI_IPU_FORMAT_RGB888
    if (fmt == MI_IPU_FORMAT_RGB888)
    {
        return pack_rgb888_tight(
            model_input_bgr, true, model_w, model_h, buffer_bytes, input_buffer, "RGB888");
    }
#endif
#ifdef MI_IPU_FORMAT_BGR888
    if (fmt == MI_IPU_FORMAT_BGR888)
    {
        return pack_rgb888_tight(
            model_input_bgr, false, model_w, model_h, buffer_bytes, input_buffer, "BGR888");
    }
#endif

    if (fmt == MI_IPU_FORMAT_U8)
    {
        const std::size_t rgb_bytes =
            static_cast<std::size_t>(model_w) * static_cast<std::size_t>(model_h) * 3u;
        const std::size_t gray_bytes =
            static_cast<std::size_t>(model_w) * static_cast<std::size_t>(model_h);
        if (buffer_bytes >= rgb_bytes)
        {
            return pack_rgb888_tight(
                model_input_bgr, true, model_w, model_h, buffer_bytes, input_buffer, "U8-as-RGB888");
        }
        if (buffer_bytes >= gray_bytes)
        {
            cv::Mat gray;
            cv::cvtColor(model_input_bgr, gray, cv::COLOR_BGR2GRAY);
            return pack_strided_rows(
                gray.ptr<std::uint8_t>(0),
                static_cast<int>(gray.step[0]),
                model_w,
                model_h,
                static_cast<std::uint32_t>(model_w),
                buffer_bytes,
                input_buffer,
                "U8-GRAY");
        }
    }

    std::fprintf(stderr, "unsupported input elm_format: %u\n", elm_format);
    return false;
}

} // namespace

int main(int argc, char **argv)
{
    Cli cli;
    if (!parse_cli(argc, argv, cli))
    {
        return 1;
    }

    std::printf("yolov8_detect_dump_sigmastar\n");
    std::printf("  model     : %s\n", cli.model.c_str());
    std::printf("  image-dir : %s\n", cli.image_dir.c_str());
    std::printf("  out       : %s\n", cli.out.c_str());
    std::printf("  conf      : %.4f\n", static_cast<double>(cli.conf));
    std::printf("  nms       : %.4f\n", static_cast<double>(cli.nms));
    std::printf("  max_det   : %d\n", kMaxDet);

    if (!fs::is_regular_file(cli.model))
    {
        std::fprintf(stderr, "model not found: %s\n", cli.model.c_str());
        return 1;
    }

    const fs::path image_dir = fs::absolute(cli.image_dir);
    std::vector<std::string> rel_paths;
    if (!collect_images(image_dir, rel_paths))
    {
        return 1;
    }
    if (rel_paths.empty())
    {
        std::fprintf(stderr, "no images under: %s\n", image_dir.c_str());
        return 1;
    }
    std::printf("  images    : %zu\n", rel_paths.size());

    SigmastarRuntime::Params runtime_params;
    runtime_params.model_paths = {cli.model};
    SigmastarRuntime runtime(runtime_params);
    if (!runtime.ok())
    {
        std::fprintf(stderr, "SigmastarRuntime init failed (check IPU firmware)\n");
        return 1;
    }

    SigmastarEngine::Params engine_params;
    engine_params.model_path = cli.model;
    SigmastarEngine engine(runtime, engine_params);
    if (!engine.ok())
    {
        std::fprintf(stderr, "SigmastarEngine init failed\n");
        return 1;
    }

    std::printf("%s", engine.dump_info().c_str());

    const auto &desc = engine.getInfo().desc;
    if (desc.u32InputTensorCount < 1u)
    {
        std::fprintf(stderr, "SigmastarEngine has no input tensors\n");
        return 1;
    }

    const auto &in0 = desc.astMI_InputTensorDescs[0];
    const int model_h = static_cast<int>(in0.u32TensorShape[1]);
    const int model_w = static_cast<int>(in0.u32TensorShape[2]);
    const std::size_t buffer_bytes = in0.u32BufSize;
    const std::uint32_t elm_format = static_cast<std::uint32_t>(in0.eElmFormat);
    std::printf(
        "  input elm : %s (%u), model %dx%d\n",
        elm_format_name(elm_format),
        elm_format,
        model_w,
        model_h);

    YoloV8DetectPostProcessSigmastar::Params post_params;
    post_params.model_in_w = model_w;
    post_params.model_in_h = model_h;
    post_params.conf_threshold = cli.conf;
    post_params.nms_threshold = cli.nms;
    post_params.obj_numb_max_size = kMaxDet;
    post_params.output_shapes = extract_shapes_from_MI_IPU_TensorDesc_t(
        desc.astMI_OutputTensorDescs, desc.u32OutputTensorCount);
    post_params.output_scalars = extract_fscalars_from_MI_IPU_TensorDesc_t(
        desc.astMI_OutputTensorDescs, desc.u32OutputTensorCount);

    YoloV8DetectPostProcessSigmastar processor(post_params);

    json root;
    root["meta"] = {
        {"format_version", kFormatVersion},
        {"source", "board"},
        {"model", fs::path(cli.model).filename().string()},
        {"dump_conf", cli.conf},
        {"nms", cli.nms},
        {"max_det", kMaxDet},
        {"coord_space", "original_xyxy"},
        {"preprocess", "resize"},
        {"model_in_w", model_w},
        {"model_in_h", model_h},
    };
    root["images"] = json::array();

    std::vector<std::uint8_t> input_buffer;
    std::size_t ok_count = 0;
    std::size_t fail_count = 0;

    for (std::size_t i = 0; i < rel_paths.size(); ++i)
    {
        const std::string &rel = rel_paths[i];
        const fs::path abs_path = image_dir / rel;

        cv::Mat orig = cv::imread(abs_path.string(), cv::IMREAD_COLOR);
        if (orig.empty())
        {
            std::fprintf(stderr, "[%zu/%zu] failed to read: %s\n", i + 1, rel_paths.size(), rel.c_str());
            ++fail_count;
            continue;
        }

        const int img_w = orig.cols;
        const int img_h = orig.rows;
        cv::Mat model_input_bgr;
        cv::resize(orig, model_input_bgr, cv::Size(model_w, model_h));

        if (!pack_model_input(
                model_input_bgr, model_w, model_h, elm_format, buffer_bytes, input_buffer))
        {
            std::fprintf(stderr, "[%zu/%zu] pack failed: %s\n", i + 1, rel_paths.size(), rel.c_str());
            ++fail_count;
            continue;
        }

        if (!engine.run(input_buffer.data(), input_buffer.size()))
        {
            std::fprintf(stderr, "[%zu/%zu] engine.run failed: %s\n", i + 1, rel_paths.size(), rel.c_str());
            ++fail_count;
            continue;
        }

        const auto &tensor_views = engine.getResult().tensor_views;
        if (tensor_views.empty() || !processor.run(tensor_views))
        {
            std::fprintf(stderr, "[%zu/%zu] postprocess failed: %s\n", i + 1, rel_paths.size(), rel.c_str());
            engine.releaseResult();
            ++fail_count;
            continue;
        }
        engine.releaseResult();

        auto dets = processor.getResult().detections;
        std::sort(
            dets.begin(),
            dets.end(),
            [](const auto &a, const auto &b) { return a.score > b.score; });

        const float sx = static_cast<float>(img_w) / static_cast<float>(model_w);
        const float sy = static_cast<float>(img_h) / static_cast<float>(model_h);

        json image_obj;
        image_obj["path_img"] = rel;
        image_obj["img_w"] = img_w;
        image_obj["img_h"] = img_h;
        image_obj["detections"] = json::array();

        for (const auto &det : dets)
        {
            image_obj["detections"].push_back(
                {{"class_id", det.class_id},
                 {"score", det.score},
                 {"x1", static_cast<float>(det.box.left) * sx},
                 {"y1", static_cast<float>(det.box.top) * sy},
                 {"x2", static_cast<float>(det.box.right) * sx},
                 {"y2", static_cast<float>(det.box.bottom) * sy}});
        }

        root["images"].push_back(std::move(image_obj));
        ++ok_count;

        if ((i + 1) % 50 == 0 || i + 1 == rel_paths.size())
        {
            std::printf(
                "  progress  : %zu/%zu (ok=%zu fail=%zu)\n",
                i + 1,
                rel_paths.size(),
                ok_count,
                fail_count);
        }
    }

    const fs::path out_path = fs::path(cli.out);
    std::error_code ec;
    if (!out_path.parent_path().empty())
    {
        fs::create_directories(out_path.parent_path(), ec);
    }

    std::ofstream ofs(out_path);
    if (!ofs)
    {
        std::fprintf(stderr, "failed to open out: %s\n", cli.out.c_str());
        return 1;
    }
    ofs << root.dump(2) << '\n';
    if (!ofs)
    {
        std::fprintf(stderr, "failed to write out: %s\n", cli.out.c_str());
        return 1;
    }

    std::printf(
        "done: images_ok=%zu fail=%zu written=%s\n",
        ok_count,
        fail_count,
        cli.out.c_str());
    return fail_count > 0 && ok_count == 0 ? 1 : 0;
}
