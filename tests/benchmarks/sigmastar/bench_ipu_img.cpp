/**
 * Sigmastar IPU .img 模型无关性能 benchmark（自包含，不经 SigmastarEngine）。
 *
 * 渊源:
 *   - IPU 调用路径源自 SigmaStar SDK verify 样例
 *     Pcupid_DLC173V2.3.3_0918/Linux_SDK_Release_20250918/SourceCode/sdk/verify/release_feature/source/dla/dla_show_img_info
 *     dla_show_img_info（GetOfflineModeStaticInfo / BatchMode 分支 /
 *     MMA 张量缓冲 / Invoke2·Invoke2Custom / RuntimeInfo）。
 *   - 指标与跑法对齐 OpenDLA prog_benchmark：warmup、多轮统计、
 *     bandwidth、ROM/RAM 估算；随机填充输入；并打印单模型 IO 详情。
 *
 * 用法:
 *   bench_ipu_img [options] [path]
 *
 * path:
 *   (省略)     → 扫 fixtures/integration/sigmastar 下全部 *.img
 *   model.img  → 单模型
 *   model_dir/ → 目录递归 *.img
 *
 * options:
 *   --batch N   N_BUF 模式 batch（默认 1）
 *   -h, --help
 *
 * 固定: warmup=5, loops=50, firmware=/config/dla/ipu_firmware.bin, fill=random
 */
#include "test_common/paths.hpp"

#include "mi_common_datatype.h"
#include "mi_ipu.h"
#include "mi_sys.h"
#include "mi_sys_datatype.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <getopt.h>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kDefaultBatch = 1;
constexpr int kWarmup = 5;
constexpr int kLoops = 50;
constexpr const char *kFirmware = "/config/dla/ipu_firmware.bin";
constexpr const char *kDefaultFixtureRel = "integration/sigmastar";
constexpr MI_U16 kSocId = 0;

struct Options
{
    int batch{kDefaultBatch};
    std::string path; // empty → auto fixtures
};

struct ModelStats
{
    std::string model_name;
    std::string model_path;
    const char *batch_mode{"?"};
    int batch{1};
    bool ok{false};
    double avg_ms{0};
    double min_ms{0};
    double p50_ms{0};
    double p99_ms{0};
    double avg_bw_kb{0};
    double rom_kb{0};
    double ram_kb{0};
    std::string error;
};

const char *tensor_type_name(MI_IPU_ELEMENT_FORMAT fmt)
{
    switch (fmt)
    {
    case MI_IPU_FORMAT_U8:
        return "UINT8";
    case MI_IPU_FORMAT_NV12:
        return "YUV_NV12";
    case MI_IPU_FORMAT_INT16:
        return "INT16";
    case MI_IPU_FORMAT_INT32:
        return "INT32";
    case MI_IPU_FORMAT_INT8:
        return "INT8";
    case MI_IPU_FORMAT_FP32:
        return "FLOAT32";
    case MI_IPU_FORMAT_UNKNOWN:
        return "UNKNOWN";
    case MI_IPU_FORMAT_ARGB8888:
        return "BGRA";
    case MI_IPU_FORMAT_ABGR8888:
        return "RGBA";
    case MI_IPU_FORMAT_GRAY:
        return "GRAY";
    case MI_IPU_FORMAT_COMPLEX64:
        return "COMPLEX64";
    default:
        return "NOTYPE";
    }
}

void print_tensor_desc(const char *kind, MI_U32 idx, const MI_IPU_TensorDesc_t &td)
{
    std::printf("%s(%u):\n", kind, idx);
    std::printf("    name:\t%s\n", td.name);
    std::printf("    dtype:\t%s\n", tensor_type_name(td.eElmFormat));
    std::printf("    shape:\t[");
    for (MI_U32 d = 0; d < td.u32TensorDim; ++d)
    {
        std::printf("%u%s", td.u32TensorShape[d], (d + 1 < td.u32TensorDim) ? ", " : "");
    }
    std::printf("]\n");
    std::printf("    size:\t%d\n", td.s32AlignedBufSize);
    if (td.eLayoutType == E_IPU_LAYOUT_TYPE_NCHW)
    {
        std::printf("    layout:\tNCHW\n");
    }
    if (td.eElmFormat == MI_IPU_FORMAT_INT16)
    {
        std::printf("    quantization:\t(%g, %lld)\n",
                    static_cast<double>(td.fScalar),
                    static_cast<long long>(td.s64ZeroPoint));
    }
}

void print_model_details(
    const std::string &model_path,
    MI_U32 channel_id,
    const MI_IPU_OfflineModelStaticInfo_t &offline,
    const MI_IPU_SubNet_InputOutputDesc_t &desc,
    int batch)
{
    std::printf("%s(%u):\n", model_path.c_str(), channel_id);
    std::printf(
        "Variable size: %.6f MB\n",
        offline.u32VariableBufferSize / 1024.0 / 1024.0);
    std::printf(
        "Work mode:     %s\n",
        offline.eIpuWorkMode == E_IPU_IPU_WORK_MODE_SINGLECORE ? "single_core" : "multicore");

    if (offline.eBatchMode == E_IPU_BATCH_N_BUF_MODE)
    {
        std::printf("Batch mode:    n_buf\n");
        std::printf("Setting batch: %u\n", offline.au32BatchNumTypes[0]);
        std::printf("Suggest batch: [");
        const MI_U32 max_batch = offline.au32BatchNumTypes[1];
        const MI_U32 steps = (max_batch == 1) ? 1u : static_cast<MI_U32>(std::log2(max_batch));
        MI_U32 cur = 1;
        for (MI_U32 i = 0; i < steps; ++i)
        {
            std::printf("%u%s", cur, (i + 1 < steps) ? ", " : "");
            cur *= 2;
        }
        std::printf("]\n");
        std::printf("Run batch:     %d\n", batch);
    }
    else
    {
        std::printf("Batch mode:    one_buf\n");
        std::printf("Support batch: [");
        for (MI_U32 i = 0; i < offline.u32TotalBatchNumTypes; ++i)
        {
            std::printf(
                "%u%s",
                offline.au32BatchNumTypes[i],
                (i + 1 < offline.u32TotalBatchNumTypes) ? ", " : "");
        }
        std::printf("]\n");
        std::printf("Run batch:     %d\n", batch);
    }

    for (MI_U32 i = 0; i < desc.u32InputTensorCount; ++i)
    {
        print_tensor_desc("Input", i, desc.astMI_InputTensorDescs[i]);
    }
    for (MI_U32 i = 0; i < desc.u32OutputTensorCount; ++i)
    {
        print_tensor_desc("Output", i, desc.astMI_OutputTensorDescs[i]);
    }
    std::printf("\n");
}

void show_usage(const char *exe)
{
    std::printf(
        "Usage: %s [options] [path]\n"
        "\n"
        "  path omitted     scan fixtures/%s for *.img\n"
        "  path = file.img  benchmark one model\n"
        "  path = directory recursive *.img\n"
        "\n"
        "Options:\n"
        "  --batch N   batch size for N_BUF mode (default %d)\n"
        "  -h, --help\n"
        "\n"
        "Fixed: warmup=%d  loops=%d  firmware=%s  fill=random\n",
        exe,
        kDefaultFixtureRel,
        kDefaultBatch,
        kWarmup,
        kLoops,
        kFirmware);
}

bool parse_args(int argc, char **argv, Options &opt)
{
    static const option long_opts[] = {
        {"batch", required_argument, nullptr, 'b'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int c = 0;
    while ((c = getopt_long(argc, argv, "b:h", long_opts, nullptr)) != -1)
    {
        switch (c)
        {
        case 'b':
            opt.batch = std::atoi(optarg);
            break;
        case 'h':
            show_usage(argv[0]);
            std::exit(0);
        default:
            show_usage(argv[0]);
            return false;
        }
    }

    if (opt.batch < 1)
    {
        std::fprintf(stderr, "invalid --batch\n");
        return false;
    }

    if (optind < argc)
    {
        opt.path = argv[optind];
    }
    return true;
}

std::string resolve_user_path(const std::string &path)
{
    fs::path p(path);
    if (p.is_relative())
    {
        p = fs::current_path() / p;
    }
    std::error_code ec;
    const fs::path canonical = fs::weakly_canonical(p, ec);
    return ec ? p.string() : canonical.string();
}

void collect_img_files(const fs::path &dir, std::vector<std::string> &out)
{
    std::error_code ec;
    const fs::recursive_directory_iterator end;
    for (fs::recursive_directory_iterator it(dir, ec); it != end; it.increment(ec))
    {
        if (ec)
        {
            break;
        }
        if (!it->is_regular_file())
        {
            continue;
        }
        if (it->path().extension() == ".img")
        {
            out.push_back(it->path().string());
        }
    }
    std::sort(out.begin(), out.end());
}

bool resolve_models(const Options &opt, std::vector<std::string> &models, std::string &mode_desc)
{
    models.clear();
    if (opt.path.empty())
    {
        const fs::path dir = percept::test::fixture_dir(kDefaultFixtureRel);
        if (!fs::is_directory(dir))
        {
            std::fprintf(
                stderr,
                "auto mode: fixture dir not found (expected fixtures/%s)\n",
                kDefaultFixtureRel);
            return false;
        }
        collect_img_files(dir, models);
        mode_desc = "auto (" + dir.string() + ")";
    }
    else
    {
        const std::string resolved = resolve_user_path(opt.path);
        std::error_code ec;
        if (fs::is_regular_file(resolved, ec))
        {
            if (fs::path(resolved).extension() != ".img")
            {
                std::fprintf(stderr, "not a .img file: %s\n", resolved.c_str());
                return false;
            }
            models.push_back(resolved);
            mode_desc = "file";
        }
        else if (fs::is_directory(resolved, ec))
        {
            collect_img_files(resolved, models);
            mode_desc = "dir (" + resolved + ")";
        }
        else
        {
            std::fprintf(stderr, "path not found: %s\n", resolved.c_str());
            return false;
        }
    }

    if (models.empty())
    {
        std::fprintf(stderr, "no .img models to benchmark\n");
        return false;
    }
    return true;
}

long long file_size_bytes(const std::string &path)
{
    std::error_code ec;
    const auto sz = fs::file_size(path, ec);
    return ec ? 0 : static_cast<long long>(sz);
}

std::uint64_t output_tensor_bytes(const MI_IPU_SubNet_InputOutputDesc_t &desc)
{
    std::uint64_t total = 0;
    for (MI_U32 i = 0; i < desc.u32OutputTensorCount; ++i)
    {
        total += desc.astMI_OutputTensorDescs[i].u32BufSize;
    }
    return total;
}

void fill_buffer_random(void *data, std::size_t bytes, std::mt19937 &rng)
{
    if (data == nullptr || bytes == 0)
    {
        return;
    }
    auto *p = static_cast<std::uint8_t *>(data);
    std::uniform_int_distribution<int> dist(0, 255);
    for (std::size_t i = 0; i < bytes; ++i)
    {
        p[i] = static_cast<std::uint8_t>(dist(rng));
    }
}

bool tensor_buffer_malloc(MI_IPU_Tensor_t *tensor, MI_U32 buf_size)
{
    MI_PHY phy = 0;
    void *virt = nullptr;
    if (MI_SYS_MMA_Alloc(kSocId, nullptr, buf_size, &phy) != MI_SUCCESS)
    {
        std::fprintf(stderr, "MMA_Alloc %u failed\n", buf_size);
        return false;
    }
    if (MI_SYS_Mmap(phy, buf_size, &virt, TRUE) != MI_SUCCESS)
    {
        MI_SYS_MMA_Free(kSocId, phy);
        std::fprintf(stderr, "Mmap %u failed\n", buf_size);
        return false;
    }
    tensor->phyTensorAddr[0] = phy;
    tensor->ptTensorData[0] = virt;
    return true;
}

void tensor_buffer_free(MI_IPU_Tensor_t *tensor, MI_U32 buf_size)
{
    if (tensor->ptTensorData[0] != nullptr)
    {
        MI_SYS_Munmap(tensor->ptTensorData[0], buf_size);
        tensor->ptTensorData[0] = nullptr;
    }
    if (tensor->phyTensorAddr[0] != 0)
    {
        MI_SYS_MMA_Free(kSocId, tensor->phyTensorAddr[0]);
        tensor->phyTensorAddr[0] = 0;
    }
}

bool get_tensor_buffers(
    const MI_IPU_SubNet_InputOutputDesc_t &desc,
    MI_IPU_BatchInvokeParam_t &invoke,
    MI_IPU_BatchMode_e batch_mode)
{
    if (batch_mode == E_IPU_BATCH_N_BUF_MODE)
    {
        for (MI_U32 b = 0; b < invoke.u32BatchN; ++b)
        {
            for (MI_U32 i = 0; i < desc.u32InputTensorCount; ++i)
            {
                const MI_U32 idx = b * desc.u32InputTensorCount + i;
                if (!tensor_buffer_malloc(
                        &invoke.astArrayTensors[idx],
                        static_cast<MI_U32>(desc.astMI_InputTensorDescs[i].s32AlignedBufSize)))
                {
                    return false;
                }
            }
        }
        const MI_U32 input_num = invoke.u32BatchN * desc.u32InputTensorCount;
        for (MI_U32 b = 0; b < invoke.u32BatchN; ++b)
        {
            for (MI_U32 o = 0; o < desc.u32OutputTensorCount; ++o)
            {
                const MI_U32 idx = input_num + b * desc.u32OutputTensorCount + o;
                if (!tensor_buffer_malloc(
                        &invoke.astArrayTensors[idx],
                        static_cast<MI_U32>(desc.astMI_OutputTensorDescs[o].s32AlignedBufSize)))
                {
                    return false;
                }
            }
        }
        return true;
    }

    for (MI_U32 i = 0; i < desc.u32InputTensorCount; ++i)
    {
        const MI_U32 bytes = desc.astMI_InputTensorDescs[i].u32BufSize * invoke.u32BatchN;
        if (!tensor_buffer_malloc(&invoke.astArrayTensors[i], bytes))
        {
            return false;
        }
    }
    for (MI_U32 o = 0; o < desc.u32OutputTensorCount; ++o)
    {
        const MI_U32 bytes = desc.astMI_OutputTensorDescs[o].u32BufSize * invoke.u32BatchN;
        if (!tensor_buffer_malloc(
                &invoke.astArrayTensors[desc.u32InputTensorCount + o], bytes))
        {
            return false;
        }
    }
    return true;
}

void put_tensor_buffers(
    const MI_IPU_SubNet_InputOutputDesc_t &desc,
    MI_IPU_BatchInvokeParam_t &invoke,
    MI_IPU_BatchMode_e batch_mode)
{
    if (batch_mode == E_IPU_BATCH_N_BUF_MODE)
    {
        for (MI_U32 b = 0; b < invoke.u32BatchN; ++b)
        {
            for (MI_U32 i = 0; i < desc.u32InputTensorCount; ++i)
            {
                const MI_U32 idx = b * desc.u32InputTensorCount + i;
                tensor_buffer_free(
                    &invoke.astArrayTensors[idx],
                    static_cast<MI_U32>(desc.astMI_InputTensorDescs[i].s32AlignedBufSize));
            }
        }
        const MI_U32 input_num = invoke.u32BatchN * desc.u32InputTensorCount;
        for (MI_U32 b = 0; b < invoke.u32BatchN; ++b)
        {
            for (MI_U32 o = 0; o < desc.u32OutputTensorCount; ++o)
            {
                const MI_U32 idx = input_num + b * desc.u32OutputTensorCount + o;
                tensor_buffer_free(
                    &invoke.astArrayTensors[idx],
                    static_cast<MI_U32>(desc.astMI_OutputTensorDescs[o].s32AlignedBufSize));
            }
        }
        return;
    }

    for (MI_U32 i = 0; i < desc.u32InputTensorCount; ++i)
    {
        const MI_U32 bytes = desc.astMI_InputTensorDescs[i].u32BufSize * invoke.u32BatchN;
        tensor_buffer_free(&invoke.astArrayTensors[i], bytes);
    }
    for (MI_U32 o = 0; o < desc.u32OutputTensorCount; ++o)
    {
        const MI_U32 bytes = desc.astMI_OutputTensorDescs[o].u32BufSize * invoke.u32BatchN;
        tensor_buffer_free(&invoke.astArrayTensors[desc.u32InputTensorCount + o], bytes);
    }
}

bool fill_invoke_inputs(
    const MI_IPU_SubNet_InputOutputDesc_t &desc,
    MI_IPU_BatchInvokeParam_t &invoke,
    MI_IPU_BatchMode_e batch_mode,
    std::mt19937 &rng)
{
    if (batch_mode == E_IPU_BATCH_N_BUF_MODE)
    {
        for (MI_U32 b = 0; b < invoke.u32BatchN; ++b)
        {
            for (MI_U32 i = 0; i < desc.u32InputTensorCount; ++i)
            {
                const MI_U32 idx = b * desc.u32InputTensorCount + i;
                const MI_U32 bytes =
                    static_cast<MI_U32>(desc.astMI_InputTensorDescs[i].s32AlignedBufSize);
                void *data = invoke.astArrayTensors[idx].ptTensorData[0];
                fill_buffer_random(data, bytes, rng);
                if (data != nullptr && MI_SYS_FlushInvCache(data, bytes) != MI_SUCCESS)
                {
                    std::fprintf(stderr, "FlushInvCache failed\n");
                    return false;
                }
            }
        }
        return true;
    }

    for (MI_U32 i = 0; i < desc.u32InputTensorCount; ++i)
    {
        const MI_U32 bytes = desc.astMI_InputTensorDescs[i].u32BufSize * invoke.u32BatchN;
        void *data = invoke.astArrayTensors[i].ptTensorData[0];
        fill_buffer_random(data, bytes, rng);
        if (data != nullptr && MI_SYS_FlushInvCache(data, bytes) != MI_SUCCESS)
        {
            std::fprintf(stderr, "FlushInvCache failed\n");
            return false;
        }
    }
    return true;
}

double percentile_ms(std::vector<double> sorted, double q)
{
    if (sorted.empty())
    {
        return 0.0;
    }
    std::sort(sorted.begin(), sorted.end());
    const double idx = q * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

ModelStats bench_one_model(const std::string &model_path, const Options &opt)
{
    ModelStats stats;
    stats.model_path = model_path;
    stats.model_name = fs::path(model_path).filename().string();

    std::vector<char> model_cstr(model_path.begin(), model_path.end());
    model_cstr.push_back('\0');
    std::vector<char> firmware_cstr(kFirmware, kFirmware + std::strlen(kFirmware) + 1);

    MI_IPU_OfflineModelStaticInfo_t offline{};
    if (MI_IPU_GetOfflineModeStaticInfo(nullptr, model_cstr.data(), &offline) != MI_SUCCESS)
    {
        stats.error = "GetOfflineModeStaticInfo failed";
        return stats;
    }

    const bool n_buf = (offline.eBatchMode == E_IPU_BATCH_N_BUF_MODE);
    stats.batch_mode = n_buf ? "N_BUF" : "ONE_BUF";
    int batch = opt.batch;
    if (!n_buf)
    {
        batch = static_cast<int>(offline.au32BatchNumTypes[0]);
        if (batch < 1)
        {
            batch = 1;
        }
    }
    stats.batch = batch;

    MI_IPU_DevAttr_t dev_attr{};
    dev_attr.u32MaxVariableBufSize = offline.u32VariableBufferSize;
    dev_attr.u32YUV420_W_Pitch_Alignment = 16;
    dev_attr.u32YUV420_H_Pitch_Alignment = 2;
    dev_attr.u32XRGB_W_Pitch_Alignment = 16;

    if (MI_IPU_CreateDevice(&dev_attr, nullptr, firmware_cstr.data(), 0) != MI_SUCCESS)
    {
        stats.error = "CreateDevice failed (check firmware)";
        return stats;
    }

    MI_IPUChnAttr_t chn_attr{};
    if (n_buf)
    {
        chn_attr.u32InputBufDepth = 1;
        chn_attr.u32OutputBufDepth = 1;
    }
    chn_attr.u32BatchMax = static_cast<MI_U32>(batch);

    MI_U32 channel_id = 0;
    if (MI_IPU_CreateCHN(&channel_id, &chn_attr, nullptr, model_cstr.data()) != MI_SUCCESS)
    {
        MI_IPU_DestroyDevice();
        stats.error = "CreateCHN failed";
        return stats;
    }

    MI_IPU_SubNet_InputOutputDesc_t desc{};
    if (MI_IPU_GetInOutTensorDesc(channel_id, &desc) != MI_SUCCESS)
    {
        MI_IPU_DestroyCHN(channel_id);
        MI_IPU_DestroyDevice();
        stats.error = "GetInOutTensorDesc failed";
        return stats;
    }

    print_model_details(model_path, channel_id, offline, desc, batch);

    MI_IPU_BatchInvokeParam_t invoke{};
    invoke.u32BatchN = static_cast<MI_U32>(batch);
    invoke.u32IpuAffinity = 0;

    bool tensors_from_api = false;
    bool tensors_from_mma = false;

    auto cleanup = [&]() {
        if (tensors_from_api)
        {
            MI_IPU_PutInputTensors2(channel_id, &invoke);
            MI_IPU_PutOutputTensors2(channel_id, &invoke);
        }
        if (tensors_from_mma)
        {
            put_tensor_buffers(desc, invoke, offline.eBatchMode);
        }
        MI_IPU_DestroyCHN(channel_id);
        MI_IPU_DestroyDevice();
    };

    if (n_buf)
    {
        if (MI_IPU_GetInputTensors2(channel_id, &invoke) != MI_SUCCESS)
        {
            cleanup();
            stats.error = "GetInputTensors2 failed";
            return stats;
        }
        if (MI_IPU_GetOutputTensors2(channel_id, &invoke) != MI_SUCCESS)
        {
            MI_IPU_PutInputTensors2(channel_id, &invoke);
            cleanup();
            stats.error = "GetOutputTensors2 failed";
            return stats;
        }
        tensors_from_api = true;
    }
    else
    {
        if (!get_tensor_buffers(desc, invoke, offline.eBatchMode))
        {
            cleanup();
            stats.error = "tensor buffer alloc failed";
            return stats;
        }
        tensors_from_mma = true;
    }

    std::mt19937 rng(42);
    if (!fill_invoke_inputs(desc, invoke, offline.eBatchMode, rng))
    {
        cleanup();
        stats.error = "fill inputs failed";
        return stats;
    }

    auto do_invoke = [&](MI_IPU_RuntimeInfo_t &runtime) -> bool {
        std::memset(&runtime, 0, sizeof(runtime));
        const MI_S32 ret = n_buf ? MI_IPU_Invoke2(channel_id, &invoke, &runtime)
                                 : MI_IPU_Invoke2Custom(channel_id, &invoke, &runtime);
        return ret == MI_SUCCESS;
    };

    for (int i = 0; i < kWarmup; ++i)
    {
        MI_IPU_RuntimeInfo_t runtime{};
        if (!do_invoke(runtime))
        {
            cleanup();
            stats.error = "warmup Invoke failed";
            return stats;
        }
    }

    std::vector<double> times_ms;
    times_ms.reserve(static_cast<std::size_t>(kLoops));
    double bw_kb_sum = 0.0;

    for (int i = 0; i < kLoops; ++i)
    {
        MI_IPU_RuntimeInfo_t runtime{};
        const auto t0 = Clock::now();
        if (!do_invoke(runtime))
        {
            cleanup();
            stats.error = "Invoke failed";
            return stats;
        }
        const auto t1 = Clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);

        bw_kb_sum +=
            static_cast<double>(batch) * static_cast<double>(runtime.u64BandWidth) / 1024.0;
    }

    const double sum =
        std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    stats.avg_ms = sum / static_cast<double>(times_ms.size());
    stats.min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    stats.p50_ms = percentile_ms(times_ms, 0.50);
    stats.p99_ms = percentile_ms(times_ms, 0.99);
    stats.avg_bw_kb = bw_kb_sum / static_cast<double>(times_ms.size());

    const long long rom = file_size_bytes(model_path);
    stats.rom_kb = static_cast<double>(rom) / 1024.0;
    const std::uint64_t out_bytes =
        static_cast<std::uint64_t>(batch) * output_tensor_bytes(desc);
    stats.ram_kb =
        (static_cast<double>(rom) + static_cast<double>(offline.u32VariableBufferSize) +
         static_cast<double>(out_bytes)) /
        1024.0;
    stats.ok = true;

    cleanup();
    return stats;
}

void print_report(const Options &opt, const std::string &mode_desc, const std::vector<ModelStats> &rows)
{
    std::printf("bench_ipu_img\n");
    std::printf("  mode    : %s\n", mode_desc.c_str());
    std::printf(
        "  warmup  : %d  loops: %d  batch: %d  fill: random\n",
        kWarmup,
        kLoops,
        opt.batch);
    std::printf("  firmware: %s\n\n", kFirmware);

    std::printf(
        "%-48s %-8s %5s %9s %9s %9s %9s %12s %10s %10s\n",
        "model",
        "mode",
        "batch",
        "avg(ms)",
        "p50",
        "p99",
        "min",
        "bw(KB)",
        "ROM(KB)",
        "RAM(KB)");

    for (const auto &r : rows)
    {
        if (!r.ok)
        {
            std::printf("%-48s FAILED: %s\n", r.model_name.c_str(), r.error.c_str());
            continue;
        }
        std::printf(
            "%-48s %-8s %5d %9.3f %9.3f %9.3f %9.3f %12.1f %10.1f %10.1f\n",
            r.model_name.c_str(),
            r.batch_mode,
            r.batch,
            r.avg_ms,
            r.p50_ms,
            r.p99_ms,
            r.min_ms,
            r.avg_bw_kb,
            r.rom_kb,
            r.ram_kb);
    }
}

} // namespace

int main(int argc, char **argv)
{
    Options opt;
    if (!parse_args(argc, argv, opt))
    {
        return 1;
    }

    std::vector<std::string> models;
    std::string mode_desc;
    if (!resolve_models(opt, models, mode_desc))
    {
        return 1;
    }

    if (MI_SYS_Init(kSocId) != MI_SUCCESS)
    {
        std::fprintf(stderr, "MI_SYS_Init failed\n");
        return 1;
    }

    std::vector<ModelStats> rows;
    rows.reserve(models.size());
    int ok_count = 0;
    for (const auto &model : models)
    {
        std::printf("--- %s ---\n", fs::path(model).filename().string().c_str());
        ModelStats s = bench_one_model(model, opt);
        if (s.ok)
        {
            ++ok_count;
        }
        else
        {
            std::fprintf(stderr, "  error: %s\n", s.error.c_str());
        }
        rows.push_back(std::move(s));
    }

    std::printf("\n");
    print_report(opt, mode_desc, rows);

    MI_SYS_Exit(kSocId);
    return ok_count > 0 ? 0 : 1;
}
