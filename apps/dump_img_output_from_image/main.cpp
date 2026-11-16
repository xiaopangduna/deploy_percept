/**
 * Sigmastar：单张图 → .img 推理 → 一个 NPZ。
 *
 * 用法：
 *   dump_img_output_from_image --model <path.img> --image <path>
 *       [--out dump_board.npz] [-h]
 *
 * 输入格式由 .img 的 eElmFormat 决定（GRAY / RGB / BGR / BGRA / RGBA / NV12）。
 * 图已是模型 H×W 时不再缩放；否则 cv::resize INTER_LINEAR。
 * 每个输出写成 float32：元素 × fScalar，元素个数为 shape 乘积。
 * 轴顺序保持 IPU 内存顺序，不做转置。
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <exception>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "mi_ipu.h"

#include <cnpy.h>

#include "deploy_percept/core/engine/SigmastarEngine.hpp"

namespace fs = std::filesystem;

using deploy_percept::engine::SigmastarEngine;
using deploy_percept::engine::SigmastarRuntime;
using json = nlohmann::json;

namespace
{

constexpr const char *kDefaultOut = "dump_board.npz";
constexpr std::uint32_t kDivpYuvAlignment = 16;
constexpr int kFormatVersion = 1;

struct Cli
{
    std::string model;
    std::string image;
    std::string out = kDefaultOut;
};

struct Sha256
{
    std::uint32_t s[8];
    std::uint64_t bits{0};
    std::uint8_t buf[64]{};
    std::size_t n{0};
};

enum class ReadLayout
{
    Color,
    Gray,
};

struct InputPlan
{
    ReadLayout layout = ReadLayout::Color;
    int imread_flag = cv::IMREAD_COLOR;
    std::string pack_mode;
};

void print_help(const char *argv0)
{
    std::printf(
        "Usage: %s --model <path.img> --image <path> [--out %s] [-h]\n"
        "\n"
        "  --model     Sigmastar .img (required)\n"
        "  --image     one image (required). Prefer a PNG already at model HxW\n"
        "  --out       dump NPZ path (default: %s)\n"
        "  -h, --help  show this help\n"
        "\n"
        "Reads color or gray from the .img input format. Resizes with INTER_LINEAR\n"
        "only when the image size differs from the model input. Writes one NPZ:\n"
        "meta (JSON bytes) plus output_0, output_1, ... as float32 = elem * fScalar.\n"
        "Element order is the IPU buffer order.\n",
        argv0,
        kDefaultOut,
        kDefaultOut);
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
        else if (std::strcmp(a, "--image") == 0)
        {
            const char *v = need_value(a);
            if (v == nullptr)
            {
                return false;
            }
            cli.image = v;
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
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", a);
            return false;
        }
    }
    if (cli.model.empty() || cli.image.empty())
    {
        std::fprintf(stderr, "--model and --image are required\n");
        return false;
    }
    return true;
}

std::uint32_t rotr(std::uint32_t x, std::uint32_t n)
{
    return (x >> n) | (x << (32u - n));
}

void sha256_transform(Sha256 &st, const std::uint8_t block[64])
{
    static const std::uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
    {
        w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
               (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
               static_cast<std::uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i)
    {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = st.s[0];
    std::uint32_t b = st.s[1];
    std::uint32_t c = st.s[2];
    std::uint32_t d = st.s[3];
    std::uint32_t e = st.s[4];
    std::uint32_t f = st.s[5];
    std::uint32_t g = st.s[6];
    std::uint32_t h = st.s[7];
    for (int i = 0; i < 64; ++i)
    {
        const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ (~e & g);
        const std::uint32_t t1 = h + s1 + ch + k[i] + w[i];
        const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    st.s[0] += a;
    st.s[1] += b;
    st.s[2] += c;
    st.s[3] += d;
    st.s[4] += e;
    st.s[5] += f;
    st.s[6] += g;
    st.s[7] += h;
}

void sha256_init(Sha256 &st)
{
    st.s[0] = 0x6a09e667u;
    st.s[1] = 0xbb67ae85u;
    st.s[2] = 0x3c6ef372u;
    st.s[3] = 0xa54ff53au;
    st.s[4] = 0x510e527fu;
    st.s[5] = 0x9b05688cu;
    st.s[6] = 0x1f83d9abu;
    st.s[7] = 0x5be0cd19u;
    st.bits = 0;
    st.n = 0;
}

void sha256_update(Sha256 &st, const std::uint8_t *data, std::size_t len)
{
    for (std::size_t i = 0; i < len; ++i)
    {
        st.buf[st.n++] = data[i];
        if (st.n == 64)
        {
            sha256_transform(st, st.buf);
            st.bits += 512;
            st.n = 0;
        }
    }
}

std::string sha256_final(Sha256 &st)
{
    const std::uint64_t total_bits = st.bits + static_cast<std::uint64_t>(st.n) * 8u;
    st.buf[st.n++] = 0x80;
    if (st.n > 56)
    {
        while (st.n < 64)
        {
            st.buf[st.n++] = 0;
        }
        sha256_transform(st, st.buf);
        st.n = 0;
    }
    while (st.n < 56)
    {
        st.buf[st.n++] = 0;
    }
    for (int i = 7; i >= 0; --i)
    {
        st.buf[st.n++] = static_cast<std::uint8_t>((total_bits >> (8 * i)) & 0xffu);
    }
    sha256_transform(st, st.buf);

    static const char *kHex = "0123456789abcdef";
    std::string hex(64, '0');
    for (int i = 0; i < 8; ++i)
    {
        for (int b = 0; b < 4; ++b)
        {
            const std::uint8_t byte =
                static_cast<std::uint8_t>((st.s[i] >> (24 - 8 * b)) & 0xffu);
            hex[static_cast<std::size_t>(i * 8 + b * 2)] = kHex[byte >> 4];
            hex[static_cast<std::size_t>(i * 8 + b * 2 + 1)] = kHex[byte & 0x0f];
        }
    }
    return hex;
}

std::string sha256_hex(const std::uint8_t *data, std::size_t len)
{
    Sha256 st;
    sha256_init(st);
    sha256_update(st, data, len);
    return sha256_final(st);
}

std::string sha256_file(const fs::path &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return {};
    }
    std::vector<char> buf(
        (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return sha256_hex(reinterpret_cast<const std::uint8_t *>(buf.data()), buf.size());
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
    case MI_IPU_FORMAT_INT16:
        return "INT16";
    case MI_IPU_FORMAT_INT8:
        return "INT8";
    case MI_IPU_FORMAT_FP32:
        return "FP32";
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
            "gray size mismatch: got %dx%d ch=%d expect %dx%d ch=1\n",
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
            "model input size mismatch: got %dx%d ch=%d expect %dx%d ch=3\n",
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

std::size_t elem_size(MI_IPU_ELEMENT_FORMAT fmt)
{
    switch (fmt)
    {
    case MI_IPU_FORMAT_INT16:
        return sizeof(std::int16_t);
    case MI_IPU_FORMAT_FP32:
        return sizeof(float);
    case MI_IPU_FORMAT_INT8:
    case MI_IPU_FORMAT_U8:
        return sizeof(std::uint8_t);
    default:
        return 0;
    }
}

bool dequant_output(
    const void *data,
    std::size_t byte_size,
    MI_IPU_ELEMENT_FORMAT fmt,
    float scalar,
    std::size_t numel,
    float *dst)
{
    if (data == nullptr || dst == nullptr || numel == 0)
    {
        return false;
    }
    const std::size_t need = elem_size(fmt) * numel;
    if (need == 0 || byte_size < need)
    {
        return false;
    }
    if (fmt == MI_IPU_FORMAT_INT16)
    {
        const auto *src = static_cast<const std::int16_t *>(data);
        for (std::size_t i = 0; i < numel; ++i)
        {
            dst[i] = static_cast<float>(src[i]) * scalar;
        }
        return true;
    }
    if (fmt == MI_IPU_FORMAT_FP32)
    {
        const auto *src = static_cast<const float *>(data);
        for (std::size_t i = 0; i < numel; ++i)
        {
            dst[i] = src[i] * scalar;
        }
        return true;
    }
    if (fmt == MI_IPU_FORMAT_INT8)
    {
        const auto *src = static_cast<const std::int8_t *>(data);
        for (std::size_t i = 0; i < numel; ++i)
        {
            dst[i] = static_cast<float>(src[i]) * scalar;
        }
        return true;
    }
    if (fmt == MI_IPU_FORMAT_U8)
    {
        const auto *src = static_cast<const std::uint8_t *>(data);
        for (std::size_t i = 0; i < numel; ++i)
        {
            dst[i] = static_cast<float>(src[i]) * scalar;
        }
        return true;
    }
    return false;
}

} // namespace

int main(int argc, char **argv)
{
    Cli cli;
    if (!parse_cli(argc, argv, cli))
    {
        print_help(argv[0]);
        return 1;
    }

    std::printf("dump_img_output_from_image\n");
    std::printf("  model : %s\n", cli.model.c_str());
    std::printf("  image : %s\n", cli.image.c_str());
    std::printf("  out   : %s\n", cli.out.c_str());

    if (!fs::is_regular_file(cli.model))
    {
        std::fprintf(stderr, "model not found: %s\n", cli.model.c_str());
        return 1;
    }
    if (!fs::is_regular_file(cli.image))
    {
        std::fprintf(stderr, "image not found: %s\n", cli.image.c_str());
        return 1;
    }

    const std::string input_sha = sha256_file(cli.image);
    if (input_sha.empty())
    {
        std::fprintf(stderr, "failed to read image file: %s\n", cli.image.c_str());
        return 1;
    }

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
    if (in0.u32TensorDim < 3u)
    {
        std::fprintf(
            stderr,
            "input tensor rank %u, expect shape[1]=H shape[2]=W\n",
            in0.u32TensorDim);
        return 1;
    }
    const int model_h = static_cast<int>(in0.u32TensorShape[1]);
    const int model_w = static_cast<int>(in0.u32TensorShape[2]);
    const std::size_t buffer_bytes = in0.u32BufSize;
    const std::uint32_t elm_format = static_cast<std::uint32_t>(in0.eElmFormat);

    InputPlan plan;
    if (!resolve_input_plan(elm_format, model_w, model_h, buffer_bytes, plan))
    {
        return 1;
    }
    std::printf(
        "  input elm : %u (%s), HxW=%dx%d, buf=%zu\n",
        elm_format,
        elm_format_name(elm_format),
        model_h,
        model_w,
        buffer_bytes);
    std::printf("  pack mode : %s\n", plan.pack_mode.c_str());

    cv::Mat image = cv::imread(cli.image, plan.imread_flag);
    if (image.empty())
    {
        std::fprintf(stderr, "failed to decode image: %s\n", cli.image.c_str());
        return 1;
    }
    const int image_w = image.cols;
    const int image_h = image.rows;
    bool resized = false;
    if (image.cols != model_w || image.rows != model_h)
    {
        cv::Mat scaled;
        cv::resize(image, scaled, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
        image = scaled;
        resized = true;
    }
    std::printf(
        "  resized   : %s (%dx%d -> %dx%d)\n",
        resized ? "yes" : "no",
        image_w,
        image_h,
        model_w,
        model_h);

    std::vector<std::uint8_t> input_buffer;
    bool packed = false;
    if (plan.layout == ReadLayout::Gray)
    {
        packed = pack_gray_tight(image, model_w, model_h, buffer_bytes, input_buffer);
    }
    else
    {
        packed = pack_model_input(
            image, model_w, model_h, elm_format, buffer_bytes, input_buffer);
    }
    if (!packed || input_buffer.size() != buffer_bytes)
    {
        std::fprintf(stderr, "failed to pack model input\n");
        return 1;
    }
    const std::string packed_sha = sha256_hex(input_buffer.data(), input_buffer.size());

    if (!engine.run(input_buffer.data(), input_buffer.size()))
    {
        std::fprintf(stderr, "SigmastarEngine run failed\n");
        return 1;
    }
    const auto &views = engine.getResult().tensor_views;
    if (views.size() != desc.u32OutputTensorCount)
    {
        std::fprintf(
            stderr,
            "output view count %zu != desc %u\n",
            views.size(),
            desc.u32OutputTensorCount);
        return 1;
    }

    json meta;
    meta["format_version"] = kFormatVersion;
    meta["source"] = "board";
    meta["model"] = fs::path(cli.model).filename().string();
    meta["image"] = fs::path(cli.image).filename().string();
    meta["input_sha256"] = input_sha;
    meta["packed_sha256"] = packed_sha;
    meta["resized"] = resized;
    meta["image_w"] = image_w;
    meta["image_h"] = image_h;
    meta["model_w"] = model_w;
    meta["model_h"] = model_h;
    meta["elm_format"] = elm_format;
    meta["elm_format_name"] = elm_format_name(elm_format);
    meta["pack_mode"] = plan.pack_mode;
    meta["input_layout_type"] = static_cast<int>(in0.eLayoutType);
    meta["input_shape"] = json::array();
    for (MI_U32 d = 0; d < in0.u32TensorDim; ++d)
    {
        meta["input_shape"].push_back(in0.u32TensorShape[d]);
    }
    meta["dequant"] = "elm * fScalar";
    meta["stored_dtype"] = "float32";
    meta["outputs"] = json::array();

    cnpy::npz_t npz;
    for (MI_U32 i = 0; i < desc.u32OutputTensorCount; ++i)
    {
        const auto &tensor = desc.astMI_OutputTensorDescs[i];
        const auto fmt = static_cast<MI_IPU_ELEMENT_FORMAT>(tensor.eElmFormat);
        std::size_t numel = 0;
        if (!tensor_numel(tensor, numel))
        {
            std::fprintf(stderr, "output[%u] invalid shape\n", i);
            return 1;
        }
        if (views[i].data == nullptr)
        {
            std::fprintf(stderr, "output[%u] empty\n", i);
            return 1;
        }
        if (tensor.s64ZeroPoint != 0)
        {
            std::fprintf(
                stderr,
                "output[%u] zero_point=%lld, dump uses elem * fScalar\n",
                i,
                static_cast<long long>(tensor.s64ZeroPoint));
        }

        std::vector<std::size_t> shape(tensor.u32TensorDim);
        for (MI_U32 d = 0; d < tensor.u32TensorDim; ++d)
        {
            shape[d] = tensor.u32TensorShape[d];
        }
        cnpy::NpyArray array(shape, sizeof(float), false);
        if (!dequant_output(
                views[i].data,
                views[i].byte_size,
                fmt,
                tensor.fScalar,
                numel,
                array.data<float>()))
        {
            std::fprintf(
                stderr,
                "output[%u] dequant failed: dtype=%s numel=%zu bytes=%zu\n",
                i,
                elm_format_name(static_cast<std::uint32_t>(fmt)),
                numel,
                views[i].byte_size);
            return 1;
        }

        const std::string key = "output_" + std::to_string(i);
        json item;
        item["key"] = key;
        item["name"] = tensor.name[0] != '\0' ? tensor.name : "";
        item["shape"] = json::array();
        for (std::size_t d : shape)
        {
            item["shape"].push_back(d);
        }
        item["src_dtype"] = elm_format_name(static_cast<std::uint32_t>(fmt));
        item["fscalar"] = tensor.fScalar;
        item["zero_point"] = static_cast<long long>(tensor.s64ZeroPoint);
        item["numel"] = numel;
        item["layout_type"] = static_cast<int>(tensor.eLayoutType);
        item["inner_most_stride"] = tensor.u32InnerMostStride;
        const std::size_t tight_stride = shape.back() * elem_size(fmt);
        if (tensor.u32InnerMostStride != 0 &&
            tight_stride != 0 &&
            tensor.u32InnerMostStride != tight_stride)
        {
            std::fprintf(
                stderr,
                "output[%u] inner_most_stride=%u tight=%zu, dump keeps a tight prefix\n",
                i,
                tensor.u32InnerMostStride,
                tight_stride);
        }
        meta["outputs"].push_back(std::move(item));
        npz[key] = std::move(array);

        std::printf("  %s numel=%zu scalar=%g\n", key.c_str(), numel, static_cast<double>(tensor.fScalar));
    }

    const std::string meta_text = meta.dump();
    const fs::path out_path(cli.out);
    if (!out_path.parent_path().empty())
    {
        fs::create_directories(out_path.parent_path());
    }
    try
    {
        const auto *meta_bytes = reinterpret_cast<const unsigned char *>(meta_text.data());
        cnpy::npz_save(
            out_path.string(),
            "meta",
            meta_bytes,
            std::vector<std::size_t>{meta_text.size()},
            "w");
        for (const auto &item : meta["outputs"])
        {
            const std::string key = item.at("key").get<std::string>();
            const cnpy::NpyArray &array = npz.at(key);
            cnpy::npz_save(
                out_path.string(),
                key,
                array.data<float>(),
                array.shape,
                "a");
        }
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "failed to write %s: %s\n", cli.out.c_str(), ex.what());
        return 1;
    }
    std::printf("done: %s\n", cli.out.c_str());
    return 0;
}
