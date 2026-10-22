/**
 * Sigmastar：检测框裁剪分类 dump（原图 + CSV → float logits CSV）
 *
 * 用法：
 *   image_classifiter_from_detect_box_sigmastar
 *     --model <path.img> --csv <path.csv>
 *     [--out dump_board.csv] [--box-scale 1.2] [-h]
 *
 * path_img 相对 CSV 所在目录。logit = INT16 × fScalar（Softmax 前）。
 * 标签映射 A：CSV class_id 0/1 → gt_id=1，2 → gt_id=2。
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "mi_ipu.h"

#include "deploy_percept/common/types.hpp"
#include "deploy_percept/core/engine/SigmastarEngine.hpp"

namespace fs = std::filesystem;

using deploy_percept::TensorDtype;
using deploy_percept::TensorView;
using deploy_percept::engine::SigmastarEngine;
using deploy_percept::engine::SigmastarRuntime;

namespace
{

constexpr const char *kDefaultOut = "dump_board.csv";
constexpr float kDefaultBoxScale = 1.2f;
constexpr int kLetterboxFill = 114;
constexpr std::uint32_t kDivpYuvAlignment = 16;
constexpr int kExpectedNumClasses = 3;

struct Cli
{
    std::string model;
    std::string csv;
    std::string out = kDefaultOut;
    float box_scale = kDefaultBoxScale;
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

void print_help(const char *argv0)
{
    std::printf(
        "Usage: %s --model <path.img> --csv <path.csv> [--out dump_board.csv] "
        "[--box-scale 1.2] [-h]\n"
        "\n"
        "  --model       Sigmastar .img (required)\n"
        "  --csv         annotation CSV (required); path_img relative to CSV dir\n"
        "  --out         dump CSV path (default: %s)\n"
        "  --box-scale   crop expand factor (default: %.1f)\n"
        "  -h, --help    show this help\n"
        "\n"
        "Dump columns: path_img,class_id,gt_id,logit_0,logit_1,logit_2,pred_id\n"
        "gt_id mapping A: CSV 0/1 -> 1, CSV 2 -> 2\n",
        argv0,
        kDefaultOut,
        static_cast<double>(kDefaultBoxScale));
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
            cli.box_scale = static_cast<float>(std::atof(v));
            if (!(cli.box_scale > 0.f))
            {
                std::fprintf(stderr, "invalid --box-scale: %s\n", v);
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
    return true;
}

/** 标签映射 A：CSV 0/1 → 1，2 → 2；其它返回 -1 */
int map_gt_id(int class_id)
{
    if (class_id == 0 || class_id == 1)
    {
        return 1;
    }
    if (class_id == 2)
    {
        return 2;
    }
    return -1;
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

bool load_csv_rows(const fs::path &csv_path, std::vector<CsvRow> &rows)
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
    if (idx_path < 0 || idx_class < 0 || idx_cx < 0 || idx_cy < 0 || idx_w < 0 || idx_h < 0)
    {
        std::fprintf(
            stderr,
            "csv missing required columns "
            "(path_img,class_id,cx,cy,w,h): %s\n",
            csv_path.c_str());
        return false;
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
        const int need = std::max(
            {idx_path, idx_class, idx_cx, idx_cy, idx_w, idx_h});
        if (static_cast<int>(fields.size()) <= need)
        {
            std::fprintf(stderr, "csv line %d: too few fields, skip\n", line_no);
            continue;
        }
        CsvRow row;
        row.path_img = fields[static_cast<std::size_t>(idx_path)];
        row.class_id = std::atoi(fields[static_cast<std::size_t>(idx_class)].c_str());
        row.cx = static_cast<float>(std::atof(fields[static_cast<std::size_t>(idx_cx)].c_str()));
        row.cy = static_cast<float>(std::atof(fields[static_cast<std::size_t>(idx_cy)].c_str()));
        row.w = static_cast<float>(std::atof(fields[static_cast<std::size_t>(idx_w)].c_str()));
        row.h = static_cast<float>(std::atof(fields[static_cast<std::size_t>(idx_h)].c_str()));
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
            "letterbox size mismatch: got %dx%d expect %dx%d\n",
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

    std::fprintf(stderr, "unsupported input elm_format: %u\n", elm_format);
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

} // namespace

int main(int argc, char **argv)
{
    Cli cli;
    if (!parse_cli(argc, argv, cli))
    {
        return 1;
    }

    std::printf("image_classifiter_from_detect_box_sigmastar\n");
    std::printf("  model     : %s\n", cli.model.c_str());
    std::printf("  csv       : %s\n", cli.csv.c_str());
    std::printf("  out       : %s\n", cli.out.c_str());
    std::printf("  box-scale : %.4f\n", static_cast<double>(cli.box_scale));

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
    if (!load_csv_rows(csv_path, rows))
    {
        return 1;
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
    const int num_classes = static_cast<int>(out0.u32TensorShape[1]);
    const float out_scalar = out0.fScalar;

    if (num_classes < kExpectedNumClasses)
    {
        std::fprintf(
            stderr,
            "model num_classes=%d < %d, cannot dump 3 logits\n",
            num_classes,
            kExpectedNumClasses);
        return 1;
    }

    std::ofstream ofs(cli.out);
    if (!ofs)
    {
        std::fprintf(stderr, "failed to open out: %s\n", cli.out.c_str());
        return 1;
    }
    ofs << "path_img,class_id,gt_id,logit_0,logit_1,logit_2,pred_id\n";

    std::vector<std::uint8_t> input_buffer;
    std::vector<float> logits;
    std::size_t ok_count = 0;
    std::size_t skip_count = 0;

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        const CsvRow &row = rows[i];
        const int gt_id = map_gt_id(row.class_id);
        if (gt_id < 0)
        {
            std::fprintf(
                stderr,
                "[%zu] skip unmapped class_id=%d path=%s\n",
                i,
                row.class_id,
                row.path_img.c_str());
            ++skip_count;
            continue;
        }

        const fs::path img_path =
            fs::path(row.path_img).is_absolute()
                ? fs::path(row.path_img)
                : (csv_dir / row.path_img);
        cv::Mat bgr = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (bgr.empty())
        {
            std::fprintf(stderr, "[%zu] failed to read: %s\n", i, img_path.c_str());
            ++skip_count;
            continue;
        }

        const cv::Mat crop =
            crop_cxcywh(bgr, row.cx, row.cy, row.w, row.h, cli.box_scale);
        const cv::Mat letterboxed =
            letterbox_yolo(crop, model_w, model_h, kLetterboxFill);

        if (!pack_model_input(
                letterboxed,
                model_w,
                model_h,
                static_cast<std::uint32_t>(in0.eElmFormat),
                buffer_bytes,
                input_buffer))
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

        const int pred_id = argmax(logits);
        ofs << row.path_img << ',' << row.class_id << ',' << gt_id << ','
            << logits[0] << ',' << logits[1] << ',' << logits[2] << ',' << pred_id
            << '\n';
        engine.releaseResult();
        ++ok_count;

        if ((i + 1u) % 100u == 0u || i + 1u == rows.size())
        {
            std::printf("  progress: %zu / %zu (ok=%zu skip=%zu)\n", i + 1u, rows.size(), ok_count, skip_count);
        }
    }

    ofs.close();
    std::printf(
        "done: wrote %zu rows to %s (skipped %zu)\n",
        ok_count,
        cli.out.c_str(),
        skip_count);
    return ok_count > 0 ? 0 : 1;
}
