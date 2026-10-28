#include "detect_session.hpp"

#include "deploy_percept/core/engine/SigmastarEngine.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessSigmastar.hpp"

#include "mi_common.h"
#include "mi_fb_datatype.h"
#include "mi_ipu.h"
#include "mi_sys.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

namespace sensor_preview
{
namespace
{

constexpr MI_U32 kSclDev = 1;
constexpr MI_U32 kSclDetectChn = 5;
constexpr int kModelW = 800;
constexpr int kModelH = 480;
constexpr std::uint32_t kRed32 = 0xFFFF0000u;
constexpr std::uint16_t kRed16 = 0xFC00u;

deploy_percept::TensorDtype dtype_from_format(MI_IPU_ELEMENT_FORMAT fmt)
{
    switch (fmt)
    {
    case MI_IPU_FORMAT_FP32:
        return deploy_percept::TensorDtype::FP32;
    case MI_IPU_FORMAT_INT16:
        return deploy_percept::TensorDtype::INT16;
    default:
        return deploy_percept::TensorDtype::INT8;
    }
}

std::vector<char> mutable_path(const std::string &path)
{
    std::vector<char> buf(path.begin(), path.end());
    buf.push_back('\0');
    return buf;
}

void map_point(float x, float y, int degree, int dst_w, int dst_h, int &out_x, int &out_y)
{
    float rx = x;
    float ry = y;
    float rw = static_cast<float>(kModelW);
    float rh = static_cast<float>(kModelH);
    if (degree == 90)
    {
        rx = static_cast<float>(kModelH) - y;
        ry = x;
        rw = static_cast<float>(kModelH);
        rh = static_cast<float>(kModelW);
    }
    else if (degree == 270)
    {
        rx = y;
        ry = static_cast<float>(kModelW) - x;
        rw = static_cast<float>(kModelH);
        rh = static_cast<float>(kModelW);
    }
    else if (degree == 180)
    {
        rx = static_cast<float>(kModelW) - x;
        ry = static_cast<float>(kModelH) - y;
    }
    out_x = static_cast<int>(rx * static_cast<float>(dst_w) / rw);
    out_y = static_cast<int>(ry * static_cast<float>(dst_h) / rh);
}

}  // namespace

struct DetectSession::Impl
{
    DetectConfig config;
    std::atomic<bool> stop{false};
    std::thread thread;

    MI_IPU_CHN channel = 0;
    bool has_device = false;
    bool has_channel = false;
    MI_IPU_SubNet_InputOutputDesc_t desc{};
    std::unique_ptr<deploy_percept::post_process::YoloV8DetectPostProcessSigmastar> post;

    int fb_fd = -1;
    unsigned char *fb = nullptr;
    std::size_t fb_size = 0;
    int fb_w = 0;
    int fb_h = 0;
    int fb_stride = 0;
    int fb_bpp = 0;

    bool open_ipu();
    bool open_fb();
    void close_fb();
    void release_ipu();
    void loop();
    void paint(const deploy_percept::DetectionResult &result);
    void put_pixel(int x, int y);
    void stroke_rect(int x0, int y0, int x1, int y1);
};

bool DetectSession::Impl::open_ipu()
{
    auto path = mutable_path(config.model_path);
    MI_IPU_OfflineModelStaticInfo_t info{};
    if (MI_IPU_GetOfflineModeStaticInfo(nullptr, path.data(), &info) != MI_SUCCESS)
    {
        std::fprintf(stderr, "detect: read model failed: %s\n", config.model_path.c_str());
        return false;
    }

    MI_IPU_DevAttr_t dev_attr{};
    dev_attr.u32MaxVariableBufSize = info.u32VariableBufferSize;
    if (MI_IPU_CreateDevice(&dev_attr, nullptr, nullptr, 0) != MI_SUCCESS)
    {
        std::fprintf(stderr, "detect: MI_IPU_CreateDevice failed\n");
        return false;
    }
    has_device = true;

    MI_IPUChnAttr_t chn_attr{};
    chn_attr.u32InputBufDepth = 0;
    chn_attr.u32OutputBufDepth = 1;
    path = mutable_path(config.model_path);
    if (MI_IPU_CreateCHN(&channel, &chn_attr, nullptr, path.data()) != MI_SUCCESS)
    {
        std::fprintf(stderr, "detect: MI_IPU_CreateCHN failed\n");
        return false;
    }
    has_channel = true;
    if (MI_IPU_GetInOutTensorDesc(channel, &desc) != MI_SUCCESS || desc.u32InputTensorCount < 1)
    {
        std::fprintf(stderr, "detect: GetInOutTensorDesc failed\n");
        return false;
    }

    const auto &input = desc.astMI_InputTensorDescs[0];
    const int model_h = static_cast<int>(input.u32TensorShape[1]);
    const int model_w = static_cast<int>(input.u32TensorShape[2]);
    std::printf("detect model %s input %dx%d outputs %u\n",
                config.model_path.c_str(), model_w, model_h, desc.u32OutputTensorCount);
    if (model_w != kModelW || model_h != kModelH)
    {
        std::fprintf(stderr, "detect: model input must be %dx%d\n", kModelW, kModelH);
        return false;
    }

    deploy_percept::post_process::YoloV8DetectPostProcessSigmastar::Params params;
    params.model_in_w = model_w;
    params.model_in_h = model_h;
    params.conf_threshold = 0.6f;
    params.nms_threshold = 0.5f;
    params.output_shapes = deploy_percept::engine::extract_shapes_from_MI_IPU_TensorDesc_t(
        desc.astMI_OutputTensorDescs, desc.u32OutputTensorCount);
    params.output_scalars = deploy_percept::engine::extract_fscalars_from_MI_IPU_TensorDesc_t(
        desc.astMI_OutputTensorDescs, desc.u32OutputTensorCount);
    post = std::make_unique<deploy_percept::post_process::YoloV8DetectPostProcessSigmastar>(params);
    return true;
}

bool DetectSession::Impl::open_fb()
{
    fb_fd = ::open("/dev/fb0", O_RDWR);
    if (fb_fd < 0)
    {
        std::fprintf(stderr, "detect: open /dev/fb0 failed: %s\n", std::strerror(errno));
        return false;
    }
    fb_fix_screeninfo fixed{};
    fb_var_screeninfo var{};
    if (ioctl(fb_fd, FBIOGET_FSCREENINFO, &fixed) < 0 || ioctl(fb_fd, FBIOGET_VSCREENINFO, &var) < 0)
    {
        std::fprintf(stderr, "detect: read framebuffer info failed\n");
        return false;
    }
    fb_w = static_cast<int>(var.xres);
    fb_h = static_cast<int>(var.yres);
    fb_stride = static_cast<int>(fixed.line_length);
    fb_bpp = static_cast<int>(var.bits_per_pixel);
    fb_size = fixed.smem_len;
    if (fb_w <= 0 || fb_h <= 0 || (fb_bpp != 16 && fb_bpp != 32))
    {
        std::fprintf(stderr, "detect: unsupported fb %dx%d bpp %d\n", fb_w, fb_h, fb_bpp);
        return false;
    }
    void *mapped = mmap(nullptr, fb_size, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
    if (mapped == MAP_FAILED)
    {
        std::fprintf(stderr, "detect: mmap framebuffer failed\n");
        fb = nullptr;
        return false;
    }
    fb = static_cast<unsigned char *>(mapped);

    MI_FB_ColorKey_t key{};
    key.bKeyEnable = TRUE;
    ioctl(fb_fd, FBIOSET_COLORKEY, &key);
    MI_FB_GlobalAlpha_t alpha{};
    alpha.u8AlphaMode = E_MI_FB_ALPHA_CHANNEL;
    alpha.u8GlobalAlpha = 0xFF;
    alpha.u16PlaneAlpha = 0xFF;
    alpha.u8Alpha0 = 0;
    alpha.u8Alpha1 = 0xFF;
    ioctl(fb_fd, FBIOSET_GLOBAL_ALPHA, &alpha);
    MI_BOOL show = TRUE;
    if (ioctl(fb_fd, FBIOSET_SHOW, &show) < 0)
    {
        std::fprintf(stderr, "detect: FBIOSET_SHOW failed\n");
        return false;
    }
    std::printf("detect overlay fb %dx%d bpp %d\n", fb_w, fb_h, fb_bpp);
    return true;
}

void DetectSession::Impl::close_fb()
{
    if (fb != nullptr)
    {
        std::memset(fb, 0, fb_size);
        munmap(fb, fb_size);
        fb = nullptr;
    }
    if (fb_fd >= 0)
    {
        MI_BOOL show = FALSE;
        ioctl(fb_fd, FBIOSET_SHOW, &show);
        ::close(fb_fd);
        fb_fd = -1;
    }
}

void DetectSession::Impl::release_ipu()
{
    post.reset();
    if (has_channel)
    {
        MI_IPU_DestroyCHN(channel);
        has_channel = false;
    }
    if (has_device)
    {
        MI_IPU_DestroyDevice();
        has_device = false;
    }
}

void DetectSession::Impl::put_pixel(int x, int y)
{
    if (x < 0 || y < 0 || x >= fb_w || y >= fb_h)
    {
        return;
    }
    unsigned char *row = fb + static_cast<std::size_t>(y) * static_cast<std::size_t>(fb_stride);
    if (fb_bpp == 32)
    {
        auto *pixels = reinterpret_cast<std::uint32_t *>(row);
        pixels[x] = kRed32;
        return;
    }
    auto *pixels = reinterpret_cast<std::uint16_t *>(row);
    pixels[x] = kRed16;
}

void DetectSession::Impl::stroke_rect(int x0, int y0, int x1, int y1)
{
    if (x1 < x0)
    {
        std::swap(x0, x1);
    }
    if (y1 < y0)
    {
        std::swap(y0, y1);
    }
    constexpr int kThickness = 3;
    for (int t = 0; t < kThickness; ++t)
    {
        for (int x = x0; x <= x1; ++x)
        {
            put_pixel(x, y0 + t);
            put_pixel(x, y1 - t);
        }
        for (int y = y0; y <= y1; ++y)
        {
            put_pixel(x0 + t, y);
            put_pixel(x1 - t, y);
        }
    }
}

void DetectSession::Impl::paint(const deploy_percept::DetectionResult &result)
{
    if (fb == nullptr)
    {
        return;
    }
    const std::size_t visible =
        static_cast<std::size_t>(fb_stride) * static_cast<std::size_t>(fb_h);
    std::memset(fb, 0, visible < fb_size ? visible : fb_size);
    for (const auto &det : result.detections)
    {
        int x0 = 0;
        int y0 = 0;
        int x1 = 0;
        int y1 = 0;
        int corners_x[4];
        int corners_y[4];
        const float xs[4] = {static_cast<float>(det.box.left), static_cast<float>(det.box.right),
                             static_cast<float>(det.box.left), static_cast<float>(det.box.right)};
        const float ys[4] = {static_cast<float>(det.box.top), static_cast<float>(det.box.top),
                             static_cast<float>(det.box.bottom), static_cast<float>(det.box.bottom)};
        for (int i = 0; i < 4; ++i)
        {
            map_point(xs[i], ys[i], config.rotate_degree, fb_w, fb_h, corners_x[i], corners_y[i]);
        }
        x0 = corners_x[0];
        y0 = corners_y[0];
        x1 = corners_x[0];
        y1 = corners_y[0];
        for (int i = 1; i < 4; ++i)
        {
            x0 = std::min(x0, corners_x[i]);
            y0 = std::min(y0, corners_y[i]);
            x1 = std::max(x1, corners_x[i]);
            y1 = std::max(y1, corners_y[i]);
        }
        stroke_rect(x0, y0, x1, y1);
    }
    fb_var_screeninfo var{};
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &var) == 0)
    {
        var.yoffset = 0;
        ioctl(fb_fd, FBIOPAN_DISPLAY, &var);
    }
}

void DetectSession::Impl::loop()
{
    MI_SYS_ChnPort_t port{};
    port.eModId = E_MI_MODULE_ID_SCL;
    port.u32DevId = kSclDev;
    port.u32ChnId = kSclDetectChn;
    port.u32PortId = 0;
    MI_S32 fd = -1;
    if (MI_SYS_GetFd(&port, &fd) != MI_SUCCESS || fd < 0)
    {
        std::fprintf(stderr, "detect: MI_SYS_GetFd failed\n");
        return;
    }

    std::uint32_t frames = 0;
    while (!stop.load())
    {
        fd_set reads;
        FD_ZERO(&reads);
        FD_SET(fd, &reads);
        timeval tv{};
        tv.tv_sec = 0;
        tv.tv_usec = 200000;
        const int ready = ::select(fd + 1, &reads, nullptr, nullptr, &tv);
        if (ready <= 0 || !FD_ISSET(fd, &reads))
        {
            continue;
        }

        MI_SYS_BufInfo_t buf{};
        MI_SYS_BUF_HANDLE handle = 0;
        if (MI_SYS_ChnOutputPortGetBuf(&port, &buf, &handle) != MI_SUCCESS ||
            buf.eBufType != E_MI_SYS_BUFDATA_FRAME)
        {
            continue;
        }

        MI_IPU_TensorVector_t inputs{};
        MI_IPU_TensorVector_t outputs{};
        inputs.u32TensorCount = desc.u32InputTensorCount;
        inputs.astArrayTensors[0].phyTensorAddr[0] = buf.stFrameData.phyAddr[0];
        inputs.astArrayTensors[0].ptTensorData[0] = buf.stFrameData.pVirAddr[0];
        if (MI_IPU_GetOutputTensors(channel, &outputs) != MI_SUCCESS)
        {
            MI_SYS_ChnOutputPortPutBuf(handle);
            continue;
        }
        if (MI_IPU_Invoke(channel, &inputs, &outputs) != MI_SUCCESS)
        {
            std::fprintf(stderr, "detect: MI_IPU_Invoke failed\n");
            MI_IPU_PutOutputTensors(channel, &outputs);
            MI_SYS_ChnOutputPortPutBuf(handle);
            continue;
        }

        std::vector<deploy_percept::TensorView> views(desc.u32OutputTensorCount);
        for (MI_U32 i = 0; i < desc.u32OutputTensorCount; ++i)
        {
            const auto &tensor = desc.astMI_OutputTensorDescs[i];
            views[i] = {outputs.astArrayTensors[i].ptTensorData[0], tensor.u32BufSize,
                        dtype_from_format(tensor.eElmFormat)};
        }
        const bool decoded = post->run(views);
        MI_IPU_PutOutputTensors(channel, &outputs);
        MI_SYS_ChnOutputPortPutBuf(handle);
        if (!decoded)
        {
            continue;
        }
        paint(post->getResult());
        ++frames;
        if (frames % 30u == 0u)
        {
            std::printf("detect frames %u boxes %zu\n", frames, post->getResult().detections.size());
        }
    }
    MI_SYS_CloseFd(fd);
}

DetectSession::~DetectSession()
{
    stop();
}

bool DetectSession::start(const DetectConfig &config)
{
    if (impl_ != nullptr)
    {
        return false;
    }
    impl_ = new Impl();
    impl_->config = config;
    if (!impl_->open_ipu() || !impl_->open_fb())
    {
        impl_->close_fb();
        impl_->release_ipu();
        delete impl_;
        impl_ = nullptr;
        return false;
    }
    impl_->stop.store(false);
    impl_->thread = std::thread(&Impl::loop, impl_);
    std::printf("detect on, rotate %d\n", config.rotate_degree);
    return true;
}

void DetectSession::stop()
{
    if (impl_ == nullptr)
    {
        return;
    }
    impl_->stop.store(true);
    if (impl_->thread.joinable())
    {
        impl_->thread.join();
    }
    impl_->close_fb();
    impl_->release_ipu();
    delete impl_;
    impl_ = nullptr;
}

}  // namespace sensor_preview
