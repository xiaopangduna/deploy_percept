/**
 * Sigmastar：分类 dump（CSV → 一个 JSON）。
 *
 * 用法：
 *   classifier_dump_sigmastar
 *     --model <path.img> --csv <path.csv>
 *     [--out <model_stem>.json] [--box-scale 1.2]
 *     [--prep letterbox|resize|imagenet] [-h]
 *
 * CSV 有 path_img,class_id 时默认整图 LetterBox。四列 cx,cy,w,h 同时存在时先裁框。
 * --prep resize 直接 INTER_LINEAR 到模型 H×W。
 * --prep imagenet 短边缩到 floor(min(H,W)/0.875)，再中心裁到模型尺寸。
 * 标准化不在这里做。转 .img 时由 input_config 决定；ONNX 侧另做。
 * path_img 相对 CSV 所在目录。logit = 元素 × fScalar（Softmax 前）。
 * JSON 先写 meta，每条成功样本追加进 samples，最后写 summary。
 * acc1 / acc5 在 summary，只比较 class_id 与 pred_id。
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "mi_ipu.h"

#include "deploy_percept/common/types.hpp"
#include "deploy_percept/core/engine/SigmastarEngine.hpp"

namespace fs = std::filesystem;

using deploy_percept::TensorDtype;
using deploy_percept::TensorView;
using deploy_percept::engine::SigmastarEngine;
using deploy_percept::engine::SigmastarRuntime;
using json = nlohmann::ordered_json;

namespace
{

constexpr int kLetterboxFill = 114;
constexpr int kFormatVersion = 2;
constexpr int kTopKMax = 5;
constexpr std::uint32_t kDivpYuvAlignment = 16;

enum class CsvMode
{
    Image,
    Box,
};

enum class SpatialPrep
{
    Letterbox,
    Resize,
    Imagenet,
};

struct Cli
{
    std::string model;
    std::string csv;
    std::string out;
    float box_scale = 0.f;
    bool box_scale_set = false;
    SpatialPrep prep = SpatialPrep::Letterbox;
};

struct CsvRow
{
    std::string path_img;
    int class_id = -1;
    float cx = 0.f;
    float cy = 0.f;
    float w = 0.f;
    float h = 0.f;
};

const char *prep_name(SpatialPrep prep)
{
    switch (prep)
    {
    case SpatialPrep::Resize:
        return "resize";
    case SpatialPrep::Imagenet:
        return "imagenet";
    case SpatialPrep::Letterbox:
    default:
        return "letterbox";
    }
}

void print_help(const char *argv0)
{
    std::printf(
        "Usage: %s --model <path.img> --csv <path.csv> [--out <model_stem>.json]\n"
        "       [--box-scale 1.2] [--prep letterbox|resize|imagenet] [-h]\n"
        "\n"
        "  --model       Sigmastar .img (required)\n"
        "  --csv         annotation CSV (required); path_img relative to CSV dir\n"
        "  --out         dump JSON path (default: <model file stem>.json)\n"
        "  --box-scale   >0 and CSV has cx,cy,w,h: crop with this expand factor\n"
        "                omitted or <=0: whole image\n"
        "  --prep        letterbox (default): keep aspect, pad 114\n"
        "                resize: INTER_LINEAR to model HxW\n"
        "                imagenet: shorter side -> floor(min(H,W)/0.875), center crop\n"
        "  -h, --help    show this help\n"
        "\n"
        "Pixels stay raw. Mean/std is chosen when converting the .img, and on ONNX.\n"
        "\n"
        "Missing cx,cy,w,h also uses the whole image.\n"
        "Writes one JSON object: meta, samples, then summary.\n"
        "logits = elem * fScalar, before softmax.\n"
        "summary.acc1/acc5 compare class_id with pred_id.\n",
        argv0);
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
        else if (std::strcmp(a, "--csv") == 0)
        {
            const char *v = need_value(a);
            if (v == nullptr)
            {
                return false;
            }
            cli.csv = v;
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
        else if (std::strcmp(a, "--box-scale") == 0)
        {
            const char *v = need_value(a);
            if (v == nullptr)
            {
                return false;
            }
            char *end = nullptr;
            cli.box_scale = std::strtof(v, &end);
            if (end == v || *end != '\0')
            {
                std::fprintf(stderr, "invalid --box-scale: %s\n", v);
                return false;
            }
            cli.box_scale_set = true;
        }
        else if (std::strcmp(a, "--prep") == 0)
        {
            const char *v = need_value(a);
            if (v == nullptr)
            {
                return false;
            }
            if (std::strcmp(v, "letterbox") == 0)
            {
                cli.prep = SpatialPrep::Letterbox;
            }
            else if (std::strcmp(v, "resize") == 0)
            {
                cli.prep = SpatialPrep::Resize;
            }
            else if (std::strcmp(v, "imagenet") == 0)
            {
                cli.prep = SpatialPrep::Imagenet;
            }
            else
            {
                std::fprintf(stderr, "invalid --prep: %s\n", v);
                return false;
            }
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", a);
            return false;
        }
    }
    if (cli.model.empty() || cli.csv.empty())
    {
        std::fprintf(stderr, "--model and --csv are required\n");
        print_help(argv[0]);
        return false;
    }
    if (cli.out.empty())
    {
        const std::string stem = fs::path(cli.model).stem().string();
        if (stem.empty())
        {
            std::fprintf(stderr, "model path has no file name: %s\n", cli.model.c_str());
            return false;
        }
        cli.out = stem + ".json";
    }
    return true;
}

std::vector<std::string> split_csv_line(const std::string &line)
{
    std::vector<std::string> fields;
    std::string cur;
    for (char c : line)
    {
        if (c == ',')
        {
            fields.push_back(cur);
            cur.clear();
        }
        else if (c != '\r')
        {
            cur.push_back(c);
        }
    }
    fields.push_back(cur);
    return fields;
}

bool load_csv_rows(const fs::path &csv_path, std::vector<CsvRow> &rows, bool &has_box)
{
    std::ifstream ifs(csv_path);
    if (!ifs)
    {
        std::fprintf(stderr, "failed to open csv: %s\n", csv_path.c_str());
        return false;
    }
    std::string line;
    if (!std::getline(ifs, line))
    {
        std::fprintf(stderr, "empty csv: %s\n", csv_path.c_str());
        return false;
    }
    const auto header = split_csv_line(line);
    int idx_path = -1;
    int idx_class = -1;
    int idx_cx = -1;
    int idx_cy = -1;
    int idx_w = -1;
    int idx_h = -1;
    for (int i = 0; i < static_cast<int>(header.size()); ++i)
    {
        const std::string &h = header[static_cast<std::size_t>(i)];
        if (h == "path_img")
        {
            idx_path = i;
        }
        else if (h == "class_id")
        {
            idx_class = i;
        }
        else if (h == "cx")
        {
            idx_cx = i;
        }
        else if (h == "cy")
        {
            idx_cy = i;
        }
        else if (h == "w")
        {
            idx_w = i;
        }
        else if (h == "h")
        {
            idx_h = i;
        }
    }
    if (idx_path < 0 || idx_class < 0)
    {
        std::fprintf(
            stderr,
            "csv missing required columns (path_img,class_id): %s\n",
            csv_path.c_str());
        return false;
    }
    const int box_cols = (idx_cx >= 0 ? 1 : 0) + (idx_cy >= 0 ? 1 : 0) +
                         (idx_w >= 0 ? 1 : 0) + (idx_h >= 0 ? 1 : 0);
    has_box = box_cols == 4;
    if (box_cols != 0 && !has_box)
    {
        std::fprintf(
            stderr,
            "csv missing some of cx,cy,w,h, using the whole image: %s\n",
            csv_path.c_str());
    }

    int line_no = 1;
    while (std::getline(ifs, line))
    {
        ++line_no;
        if (line.empty())
        {
            continue;
        }
        const auto fields = split_csv_line(line);
        const int need = has_box
                             ? std::max({idx_path, idx_class, idx_cx, idx_cy, idx_w, idx_h})
                             : std::max(idx_path, idx_class);
        if (static_cast<int>(fields.size()) <= need)
        {
            std::fprintf(stderr, "csv line %d: too few fields, skip\n", line_no);
            continue;
        }
        CsvRow row;
        row.path_img = fields[static_cast<std::size_t>(idx_path)];
        row.class_id = std::atoi(fields[static_cast<std::size_t>(idx_class)].c_str());
        if (has_box)
        {
            row.cx = static_cast<float>(std::atof(fields[static_cast<std::size_t>(idx_cx)].c_str()));
            row.cy = static_cast<float>(std::atof(fields[static_cast<std::size_t>(idx_cy)].c_str()));
            row.w = static_cast<float>(std::atof(fields[static_cast<std::size_t>(idx_w)].c_str()));
            row.h = static_cast<float>(std::atof(fields[static_cast<std::size_t>(idx_h)].c_str()));
        }
        rows.push_back(std::move(row));
    }
    return true;
}

/** 对齐 train ImageClassifierFromDetectBox.crop_cxcywh */
cv::Mat crop_cxcywh(
    const cv::Mat &img,
    float cx,
    float cy,
    float w,
    float h,
    float box_scale)
{
    const int height = img.rows;
    const int width = img.cols;
    const float abs_w = std::max(w * static_cast<float>(width) * box_scale, 1.f);
    const float abs_h = std::max(h * static_cast<float>(height) * box_scale, 1.f);
    const float abs_cx = cx * static_cast<float>(width);
    const float abs_cy = cy * static_cast<float>(height);
    int x1 = static_cast<int>(std::floor(abs_cx - abs_w / 2.f));
    int y1 = static_cast<int>(std::floor(abs_cy - abs_h / 2.f));
    int x2 = static_cast<int>(std::ceil(abs_cx + abs_w / 2.f));
    int y2 = static_cast<int>(std::ceil(abs_cy + abs_h / 2.f));
    x1 = std::max(0, std::min(x1, width - 1));
    y1 = std::max(0, std::min(y1, height - 1));
    x2 = std::max(x1 + 1, std::min(x2, width));
    y2 = std::max(y1 + 1, std::min(y2, height));
    return img(cv::Range(y1, y2), cv::Range(x1, x2)).clone();
}

/** 对齐 train LetterBox（YOLO 风格 round pad） */
cv::Mat letterbox_yolo(
    const cv::Mat &src,
    int dst_w,
    int dst_h,
    int fill)
{
    const int src_h = src.rows;
    const int src_w = src.cols;
    const float r = std::min(
        static_cast<float>(dst_h) / static_cast<float>(src_h),
        static_cast<float>(dst_w) / static_cast<float>(src_w));
    const int resize_w = static_cast<int>(std::lround(static_cast<float>(src_w) * r));
    const int resize_h = static_cast<int>(std::lround(static_cast<float>(src_h) * r));
    float dw = static_cast<float>(dst_w - resize_w) / 2.f;
    float dh = static_cast<float>(dst_h - resize_h) / 2.f;
    const int top = static_cast<int>(std::lround(dh - 0.1));
    const int bottom = static_cast<int>(std::lround(dh + 0.1));
    const int left = static_cast<int>(std::lround(dw - 0.1));
    const int right = static_cast<int>(std::lround(dw + 0.1));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(fill, fill, fill));
    // YOLO round 偶发 ±1，强制到目标尺寸
    if (padded.cols != dst_w || padded.rows != dst_h)
    {
        cv::Mat exact(dst_h, dst_w, padded.type(), cv::Scalar(fill, fill, fill));
        const int copy_w = std::min(padded.cols, dst_w);
        const int copy_h = std::min(padded.rows, dst_h);
        padded(cv::Range(0, copy_h), cv::Range(0, copy_w))
            .copyTo(exact(cv::Range(0, copy_h), cv::Range(0, copy_w)));
        return exact;
    }
    return padded;
}

/** 短边缩到 shorter，长边按比例，INTER_LINEAR。 */
cv::Mat resize_shorter_side(const cv::Mat &src, int shorter)
{
    const int src_w = src.cols;
    const int src_h = src.rows;
    if (src_w <= 0 || src_h <= 0 || shorter <= 0 || std::min(src_w, src_h) == shorter)
    {
        return src;
    }
    int new_w = 0;
    int new_h = 0;
    if (src_w < src_h)
    {
        new_w = shorter;
        new_h = std::max(
            1,
            static_cast<int>(std::lround(
                static_cast<float>(src_h) * static_cast<float>(shorter) / static_cast<float>(src_w))));
    }
    else
    {
        new_h = shorter;
        new_w = std::max(
            1,
            static_cast<int>(std::lround(
                static_cast<float>(src_w) * static_cast<float>(shorter) / static_cast<float>(src_h))));
    }
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    return dst;
}

/** 中心裁切。源图某一边小于目标时退回直接缩放到目标尺寸。 */
cv::Mat center_crop(const cv::Mat &src, int crop_w, int crop_h)
{
    if (src.cols == crop_w && src.rows == crop_h)
    {
        return src;
    }
    if (src.cols < crop_w || src.rows < crop_h)
    {
        cv::Mat scaled;
        cv::resize(src, scaled, cv::Size(crop_w, crop_h), 0, 0, cv::INTER_LINEAR);
        return scaled;
    }
    const int x = static_cast<int>(std::lround((src.cols - crop_w) / 2.0));
    const int y = static_cast<int>(std::lround((src.rows - crop_h) / 2.0));
    return src(cv::Rect(x, y, crop_w, crop_h)).clone();
}

cv::Mat fit_model_canvas(const cv::Mat &src, int model_w, int model_h, SpatialPrep prep)
{
    if (prep == SpatialPrep::Resize)
    {
        if (src.cols == model_w && src.rows == model_h)
        {
            return src;
        }
        cv::Mat scaled;
        cv::resize(src, scaled, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
        return scaled;
    }
    if (prep == SpatialPrep::Imagenet)
    {
        const int crop = std::min(model_w, model_h);
        const int shorter = static_cast<int>(std::floor(static_cast<float>(crop) / 0.875f));
        return center_crop(resize_shorter_side(src, shorter), model_w, model_h);
    }
    return letterbox_yolo(src, model_w, model_h, kLetterboxFill);
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

enum class ReadLayout
{
    Color,
    Gray,
};

/** 由模型输入描述决定整批样本的读图与打包方式。 */
struct InputPlan
{
    ReadLayout layout = ReadLayout::Color;
    int imread_flag = cv::IMREAD_COLOR;
    std::string pack_mode;
};

bool resolve_input_plan(
    std::uint32_t elm_format,
    int model_w,
    int model_h,
    std::size_t buffer_bytes,
    InputPlan &plan)
{
    if (model_w <= 0 || model_h <= 0)
    {
        std::fprintf(stderr, "invalid model input size: %dx%d\n", model_w, model_h);
        return false;
    }

    const std::size_t rgb_bytes =
        static_cast<std::size_t>(model_w) * static_cast<std::size_t>(model_h) * 3u;
    const std::size_t gray_bytes =
        static_cast<std::size_t>(model_w) * static_cast<std::size_t>(model_h);
    const auto fmt = static_cast<MI_IPU_ELEMENT_FORMAT>(elm_format);

    auto set_color = [&](const char *mode) {
        plan.layout = ReadLayout::Color;
        plan.imread_flag = cv::IMREAD_COLOR;
        plan.pack_mode = mode;
    };
    auto set_gray = [&](const char *mode) {
        plan.layout = ReadLayout::Gray;
        plan.imread_flag = cv::IMREAD_GRAYSCALE;
        plan.pack_mode = mode;
    };

    if (fmt == MI_IPU_FORMAT_GRAY)
    {
        if (buffer_bytes < gray_bytes)
        {
            std::fprintf(
                stderr,
                "GRAY input buffer too small: buf=%zu need>=%zu\n",
                buffer_bytes,
                gray_bytes);
            return false;
        }
        set_gray("GRAY (imread grayscale)");
        return true;
    }

    if (fmt == MI_IPU_FORMAT_U8)
    {
        if (buffer_bytes >= rgb_bytes)
        {
            char mode[128];
            std::snprintf(
                mode,
                sizeof(mode),
                "U8 as RGB888 (tight), buf=%zu HxWx3=%zu",
                buffer_bytes,
                rgb_bytes);
            set_color(mode);
            return true;
        }
        if (buffer_bytes >= gray_bytes)
        {
            set_gray("U8-GRAY (imread grayscale)");
            return true;
        }
        std::fprintf(
            stderr,
            "U8 input buffer too small: buf=%zu need>=%zu\n",
            buffer_bytes,
            gray_bytes);
        return false;
    }

    if (fmt == MI_IPU_FORMAT_NV12)
    {
        set_color("NV12 (imread color)");
        return true;
    }
    if (fmt == MI_IPU_FORMAT_ARGB8888)
    {
        set_color("ARGB8888 (imread color)");
        return true;
    }
    if (fmt == MI_IPU_FORMAT_ABGR8888)
    {
        set_color("ABGR8888 (imread color)");
        return true;
    }
#ifdef MI_IPU_FORMAT_RGB888
    if (fmt == MI_IPU_FORMAT_RGB888)
    {
        set_color("RGB888 (imread color)");
        return true;
    }
#endif
#ifdef MI_IPU_FORMAT_BGR888
    if (fmt == MI_IPU_FORMAT_BGR888)
    {
        set_color("BGR888 (imread color)");
        return true;
    }
#endif

    std::fprintf(
        stderr,
        "unsupported input elm_format=%u (%s), buf=%zu, HxW=%dx%d\n",
        elm_format,
        elm_format_name(elm_format),
        buffer_bytes,
        model_w,
        model_h);
    return false;
}

/** RGB/BGR：文档要求紧密排列，不做 W 对齐。to_rgb=true 时 BGR→RGB。 */
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
    // 紧密排列：row_stride = W*3（input_formats=RGB/BGR 不用对齐）
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

/** 灰度：读图阶段已是单通道，紧密排列，stride = W。 */
bool pack_gray_tight(
    const cv::Mat &gray,
    const int model_w,
    const int model_h,
    const std::size_t buffer_bytes,
    std::vector<std::uint8_t> &input_buffer)
{
    if (gray.cols != model_w || gray.rows != model_h || gray.channels() != 1)
    {
        std::fprintf(
            stderr,
            "gray letterbox mismatch: got %dx%d ch=%d expect %dx%d ch=1\n",
            gray.cols,
            gray.rows,
            gray.channels(),
            model_w,
            model_h);
        return false;
    }
    input_buffer.assign(buffer_bytes, 0);
    return pack_strided_rows(
        gray.ptr<std::uint8_t>(0),
        static_cast<int>(gray.step[0]),
        model_w,
        model_h,
        static_cast<std::uint32_t>(model_w),
        buffer_bytes,
        input_buffer,
        "GRAY");
}

/**
 * 打包缩放后的 BGR 图。灰度模型不走这里，见 ``pack_gray_tight``。
 *
 * 量化 ``input_formats`` → 常见 ``eElmFormat``：
 *   BGRA → ARGB8888（内存 BGRA，stride 16 对齐）
 *   RGBA → ABGR8888（内存 RGBA）
 *   RGB  → RGB888（新 SDK）或 U8 + buf≈H*W*3（旧 SDK）
 *   BGR  → BGR888 或 U8
 *   YUV_NV12 → NV12
 */
bool pack_model_input(
    const cv::Mat &model_input_bgr,
    const int model_w,
    const int model_h,
    const std::uint32_t elm_format,
    const std::size_t buffer_bytes,
    std::vector<std::uint8_t> &input_buffer)
{
    if (model_input_bgr.cols != model_w || model_input_bgr.rows != model_h ||
        model_input_bgr.channels() != 3)
    {
        std::fprintf(
            stderr,
            "letterbox size mismatch: got %dx%d ch=%d expect %dx%d ch=3\n",
            model_input_bgr.cols,
            model_input_bgr.rows,
            model_input_bgr.channels(),
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
        // 量化 input_formats=BGRA：SStar ARGB 内存序为 BGRA
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

    // 旧 SDK 无 RGB888 枚举时，input_formats=RGB 常落成 U8，buf≈H*W*3。
    // buf≈H*W 的灰度 U8 在 resolve_input_plan 里走 pack_gray_tight。
    if (fmt == MI_IPU_FORMAT_U8)
    {
        const std::size_t rgb_bytes =
            static_cast<std::size_t>(model_w) * static_cast<std::size_t>(model_h) * 3u;
        if (buffer_bytes >= rgb_bytes)
        {
            return pack_rgb888_tight(
                model_input_bgr, true, model_w, model_h, buffer_bytes, input_buffer, "U8-as-RGB888");
        }
    }

    std::fprintf(
        stderr,
        "unsupported input elm_format=%u (%s), buf=%zu, HxW=%dx%d\n",
        elm_format,
        elm_format_name(elm_format),
        buffer_bytes,
        model_w,
        model_h);
    return false;
}

bool dequant_logits(
    const TensorView &output,
    int num_classes,
    float scalar,
    std::vector<float> &logits)
{
    logits.assign(static_cast<std::size_t>(num_classes), 0.f);
    if (output.data == nullptr || num_classes <= 0)
    {
        return false;
    }
    if (output.dtype == TensorDtype::INT16)
    {
        const std::size_t expected =
            static_cast<std::size_t>(num_classes) * sizeof(std::int16_t);
        if (output.byte_size < expected)
        {
            return false;
        }
        const auto *data = static_cast<const std::int16_t *>(output.data);
        for (int i = 0; i < num_classes; ++i)
        {
            logits[static_cast<std::size_t>(i)] =
                static_cast<float>(data[i]) * scalar;
        }
        return true;
    }
    if (output.dtype == TensorDtype::FP32)
    {
        const std::size_t expected =
            static_cast<std::size_t>(num_classes) * sizeof(float);
        if (output.byte_size < expected)
        {
            return false;
        }
        const auto *data = static_cast<const float *>(output.data);
        for (int i = 0; i < num_classes; ++i)
        {
            logits[static_cast<std::size_t>(i)] = data[i] * scalar;
        }
        return true;
    }
    return false;
}

int argmax(const std::vector<float> &logits)
{
    int best = 0;
    for (int i = 1; i < static_cast<int>(logits.size()); ++i)
    {
        if (logits[static_cast<std::size_t>(i)] > logits[static_cast<std::size_t>(best)])
        {
            best = i;
        }
    }
    return best;
}

/** class_id 的名次：logit 更大者在前；相等时下标更小者在前，与 argmax 一致。 */
bool in_top_k(const std::vector<float> &logits, int class_id, int top_k)
{
    if (class_id < 0 || class_id >= static_cast<int>(logits.size()) || top_k <= 0)
    {
        return false;
    }
    const float class_logit = logits[static_cast<std::size_t>(class_id)];
    int better = 0;
    for (int i = 0; i < static_cast<int>(logits.size()); ++i)
    {
        if (i == class_id)
        {
            continue;
        }
        const float logit = logits[static_cast<std::size_t>(i)];
        if (logit > class_logit || (logit == class_logit && i < class_id))
        {
            ++better;
        }
    }
    return better < top_k;
}

bool logits_finite(const std::vector<float> &logits)
{
    for (float logit : logits)
    {
        if (!std::isfinite(logit))
        {
            return false;
        }
    }
    return true;
}

bool tensor_numel(const MI_IPU_TensorDesc_t &tensor, std::size_t &numel)
{
    if (tensor.u32TensorDim == 0 || tensor.u32TensorDim > MI_IPU_MAX_TENSOR_DIM)
    {
        return false;
    }
    numel = 1;
    for (MI_U32 i = 0; i < tensor.u32TensorDim; ++i)
    {
        const std::uint32_t dim = tensor.u32TensorShape[i];
        if (dim == 0 || numel > std::numeric_limits<std::size_t>::max() / dim)
        {
            return false;
        }
        numel *= dim;
    }
    return true;
}

bool ensure_parent_dir(const fs::path &path)
{
    const fs::path parent = path.parent_path();
    if (parent.empty())
    {
        return true;
    }
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec)
    {
        std::fprintf(
            stderr,
            "failed to create directory %s: %s\n",
            parent.c_str(),
            ec.message().c_str());
        return false;
    }
    return true;
}

json make_sample(
    const CsvRow &row,
    std::size_t row_index,
    CsvMode mode,
    const std::vector<float> &logits,
    int pred_id)
{
    json sample;
    sample["row"] = row_index;
    sample["path_img"] = row.path_img;
    sample["class_id"] = row.class_id;
    if (mode == CsvMode::Box)
    {
        sample["cx"] = static_cast<double>(row.cx);
        sample["cy"] = static_cast<double>(row.cy);
        sample["w"] = static_cast<double>(row.w);
        sample["h"] = static_cast<double>(row.h);
    }
    json logits_json = json::array();
    for (float logit : logits)
    {
        logits_json.push_back(static_cast<double>(logit));
    }
    sample["logits"] = std::move(logits_json);
    sample["pred_id"] = pred_id;
    return sample;
}

struct DumpStream
{
    std::ofstream ofs;
    fs::path staging;
    fs::path out_path;
    std::size_t written = 0;
};

void write_indented(std::ostream &ofs, const json &value)
{
    const std::string text = value.dump(2, ' ', false, json::error_handler_t::replace);
    for (char c : text)
    {
        ofs << c;
        if (c == '\n')
        {
            ofs << "  ";
        }
    }
}

bool open_dump(DumpStream &dump, const fs::path &out_path, const json &meta)
{
    dump.out_path = out_path;
    dump.staging = fs::path(out_path.string() + ".writing");
    dump.written = 0;
    dump.ofs.open(dump.staging);
    if (!dump.ofs)
    {
        std::fprintf(stderr, "failed to open %s\n", dump.staging.c_str());
        return false;
    }
    dump.ofs << "{\n  \"meta\": ";
    write_indented(dump.ofs, meta);
    dump.ofs << ",\n  \"samples\": [";
    dump.ofs.flush();
    if (!dump.ofs)
    {
        std::fprintf(stderr, "failed to write %s\n", dump.staging.c_str());
        dump.ofs.close();
        std::error_code ec;
        fs::remove(dump.staging, ec);
        return false;
    }
    return true;
}

bool append_sample(DumpStream &dump, const json &sample)
{
    if (dump.written > 0)
    {
        dump.ofs << ',';
    }
    dump.ofs << "\n    " << sample.dump(-1, ' ', false, json::error_handler_t::replace);
    ++dump.written;
    dump.ofs.flush();
    if (!dump.ofs)
    {
        std::fprintf(stderr, "failed to write %s\n", dump.staging.c_str());
        return false;
    }
    return true;
}

bool finish_dump(DumpStream &dump, const json &summary)
{
    if (dump.written == 0)
    {
        dump.ofs << "],\n  \"summary\": ";
    }
    else
    {
        dump.ofs << "\n  ],\n  \"summary\": ";
    }
    write_indented(dump.ofs, summary);
    dump.ofs << "\n}\n";
    dump.ofs.flush();
    dump.ofs.close();
    if (!dump.ofs)
    {
        std::fprintf(stderr, "partial dump kept at %s\n", dump.staging.c_str());
        return false;
    }

    std::error_code ec;
    fs::rename(dump.staging, dump.out_path, ec);
    if (ec)
    {
        std::fprintf(
            stderr,
            "failed to rename %s -> %s: %s\n",
            dump.staging.c_str(),
            dump.out_path.c_str(),
            ec.message().c_str());
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    Cli cli;
    if (!parse_cli(argc, argv, cli))
    {
        return 1;
    }

    std::printf("classifier_dump_sigmastar\n");
    std::printf("  model     : %s\n", cli.model.c_str());
    std::printf("  csv       : %s\n", cli.csv.c_str());
    std::printf("  out       : %s\n", cli.out.c_str());

    if (!fs::is_regular_file(cli.model))
    {
        std::fprintf(stderr, "model not found: %s\n", cli.model.c_str());
        return 1;
    }
    if (!fs::is_regular_file(cli.csv))
    {
        std::fprintf(stderr, "csv not found: %s\n", cli.csv.c_str());
        return 1;
    }

    const fs::path csv_path = fs::absolute(cli.csv);
    const fs::path csv_dir = csv_path.parent_path();

    std::vector<CsvRow> rows;
    bool has_box = false;
    if (!load_csv_rows(csv_path, rows, has_box))
    {
        return 1;
    }
    const bool crop = cli.box_scale_set && cli.box_scale > 0.f && has_box;
    const CsvMode csv_mode = crop ? CsvMode::Box : CsvMode::Image;
    if (cli.box_scale_set && cli.box_scale > 0.f && !has_box)
    {
        std::fprintf(stderr, "--box-scale > 0 but csv has no cx,cy,w,h, using the whole image\n");
    }
    std::printf("  mode      : %s\n", csv_mode == CsvMode::Box ? "box" : "image");
    if (csv_mode == CsvMode::Box)
    {
        std::printf("  box-scale : %.4f\n", static_cast<double>(cli.box_scale));
    }
    std::printf("  samples   : %zu\n", rows.size());

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
    if (desc.u32InputTensorCount < 1u || desc.u32OutputTensorCount < 1u)
    {
        std::fprintf(stderr, "SigmastarEngine missing input/output tensors\n");
        return 1;
    }

    const auto &in0 = desc.astMI_InputTensorDescs[0];
    const auto &out0 = desc.astMI_OutputTensorDescs[0];
    const int model_h = static_cast<int>(in0.u32TensorShape[1]);
    const int model_w = static_cast<int>(in0.u32TensorShape[2]);
    const std::size_t buffer_bytes = in0.u32BufSize;
    std::size_t output_numel = 0;
    if (!tensor_numel(out0, output_numel) ||
        output_numel == 0 ||
        output_numel > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        std::fprintf(stderr, "invalid classification output shape\n");
        return 1;
    }
    const int num_classes = static_cast<int>(output_numel);
    const float out_scalar = out0.fScalar;
    const int top_k = std::min(kTopKMax, num_classes);
    if (out0.s64ZeroPoint != 0)
    {
        std::fprintf(
            stderr,
            "output zero_point=%lld, logits use elem * fScalar\n",
            static_cast<long long>(out0.s64ZeroPoint));
    }
    const std::uint32_t elm_format = static_cast<std::uint32_t>(in0.eElmFormat);
    std::printf(
        "  input elm : %u (%s), HxW=%dx%d, buf=%zu\n",
        elm_format,
        elm_format_name(elm_format),
        model_h,
        model_w,
        buffer_bytes);
    InputPlan plan;
    if (!resolve_input_plan(elm_format, model_w, model_h, buffer_bytes, plan))
    {
        return 1;
    }
    std::printf("  pack mode : %s\n", plan.pack_mode.c_str());
    std::printf("  prep      : %s\n", prep_name(cli.prep));
    std::printf("  classes   : %d\n", num_classes);
    std::printf("  top-k     : %d\n", top_k);

    const fs::path out_path(cli.out);
    if (!ensure_parent_dir(out_path))
    {
        return 1;
    }

    json meta;
    meta["format_version"] = kFormatVersion;
    meta["source"] = "board";
    meta["model"] = cli.model;
    meta["csv"] = cli.csv;
    meta["mode"] = csv_mode == CsvMode::Box ? "box" : "image";
    if (csv_mode == CsvMode::Box)
    {
        meta["box_scale"] = static_cast<double>(cli.box_scale);
    }
    meta["num_classes"] = num_classes;
    meta["output_shape"] = json::array();
    for (MI_U32 d = 0; d < out0.u32TensorDim; ++d)
    {
        meta["output_shape"].push_back(out0.u32TensorShape[d]);
    }
    meta["output_fscalar"] = static_cast<double>(out_scalar);
    meta["model_h"] = model_h;
    meta["model_w"] = model_w;
    meta["prep"] = prep_name(cli.prep);
    if (cli.prep == SpatialPrep::Letterbox)
    {
        meta["letterbox_fill"] = kLetterboxFill;
    }
    meta["elm_format"] = elm_format;
    meta["elm_format_name"] = elm_format_name(elm_format);
    meta["pack_mode"] = plan.pack_mode;
    meta["top_k"] = top_k;

    DumpStream dump;
    if (!open_dump(dump, out_path, meta))
    {
        return 1;
    }

    std::vector<std::uint8_t> input_buffer;
    std::vector<float> logits;
    std::size_t ok_count = 0;
    std::size_t skip_count = 0;
    std::size_t correct1 = 0;
    std::size_t correct5 = 0;

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        const CsvRow &row = rows[i];
        if (csv_mode == CsvMode::Box &&
            !(std::isfinite(row.cx) && std::isfinite(row.cy) && std::isfinite(row.w) &&
              std::isfinite(row.h)))
        {
            std::fprintf(
                stderr,
                "[%zu] skip non-finite box path=%s\n",
                i,
                row.path_img.c_str());
            ++skip_count;
            continue;
        }

        const fs::path img_path =
            fs::path(row.path_img).is_absolute()
                ? fs::path(row.path_img)
                : (csv_dir / row.path_img);
        cv::Mat image = cv::imread(img_path.string(), plan.imread_flag);
        if (image.empty())
        {
            std::fprintf(stderr, "[%zu] failed to read: %s\n", i, img_path.c_str());
            ++skip_count;
            continue;
        }

        const cv::Mat region =
            csv_mode == CsvMode::Box
                ? crop_cxcywh(image, row.cx, row.cy, row.w, row.h, cli.box_scale)
                : image;
        const cv::Mat fitted = fit_model_canvas(region, model_w, model_h, cli.prep);

        bool packed = false;
        if (plan.layout == ReadLayout::Gray)
        {
            packed = pack_gray_tight(
                fitted, model_w, model_h, buffer_bytes, input_buffer);
        }
        else
        {
            packed = pack_model_input(
                fitted,
                model_w,
                model_h,
                elm_format,
                buffer_bytes,
                input_buffer);
        }
        if (!packed)
        {
            std::fprintf(stderr, "[%zu] pack input failed: %s\n", i, row.path_img.c_str());
            ++skip_count;
            continue;
        }

        if (!engine.run(input_buffer.data(), input_buffer.size()))
        {
            std::fprintf(stderr, "[%zu] engine.run failed: %s\n", i, row.path_img.c_str());
            ++skip_count;
            continue;
        }

        const auto &tensor_views = engine.getResult().tensor_views;
        if (tensor_views.empty() ||
            !dequant_logits(tensor_views[0], num_classes, out_scalar, logits))
        {
            std::fprintf(stderr, "[%zu] dequant failed: %s\n", i, row.path_img.c_str());
            engine.releaseResult();
            ++skip_count;
            continue;
        }

        if (!logits_finite(logits))
        {
            std::fprintf(stderr, "[%zu] non-finite logits: %s\n", i, row.path_img.c_str());
            engine.releaseResult();
            ++skip_count;
            continue;
        }

        const int pred_id = argmax(logits);
        if (!append_sample(dump, make_sample(row, i, csv_mode, logits, pred_id)))
        {
            engine.releaseResult();
            return 1;
        }
        if (pred_id == row.class_id)
        {
            ++correct1;
        }
        if (in_top_k(logits, row.class_id, top_k))
        {
            ++correct5;
        }
        engine.releaseResult();
        ++ok_count;

        if ((i + 1u) % 100u == 0u || i + 1u == rows.size())
        {
            std::printf("  progress: %zu / %zu (ok=%zu skip=%zu)\n", i + 1u, rows.size(), ok_count, skip_count);
        }
    }

    json summary;
    summary["ok"] = ok_count;
    summary["skip"] = skip_count;
    if (ok_count == 0)
    {
        summary["acc1"] = json(nullptr);
        summary["acc5"] = json(nullptr);
        std::printf("acc1: n/a\n");
        std::printf("acc5: n/a (top_k=%d)\n", top_k);
    }
    else
    {
        const double acc1 = static_cast<double>(correct1) / static_cast<double>(ok_count);
        const double acc5 = static_cast<double>(correct5) / static_cast<double>(ok_count);
        summary["acc1"] = acc1;
        summary["acc5"] = acc5;
        std::printf("acc1: %zu/%zu = %.6f\n", correct1, ok_count, acc1);
        std::printf("acc5: %zu/%zu = %.6f (top_k=%d)\n", correct5, ok_count, acc5, top_k);
    }

    if (!finish_dump(dump, summary))
    {
        return 1;
    }
    std::printf(
        "done: wrote %zu samples to %s (skipped %zu)\n",
        ok_count,
        out_path.c_str(),
        skip_count);
    return ok_count > 0 ? 0 : 1;
}
