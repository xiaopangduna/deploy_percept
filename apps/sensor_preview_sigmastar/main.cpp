/**
 * 单摄上屏：sensor pad0 → VIF → ISP → SCL → DISP → MIPI/TTL panel。
 *
 * 默认与已在板上跑通的命令一致：
 *   sensor_preview_sigmastar -c 1 -n 0 -p 1 -t 0 -r 90 -i /customer
 *
 * 90/180/270 度旋转在 SCL 完成，DISP 层保持不旋转。
 * 输入 110 或 Ctrl+C 后按绑定逆序释放。
 */
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <unistd.h>

#include "detect_session.hpp"

#include "mi_common.h"
#include "mi_disp.h"
#include "mi_isp.h"
#include "mi_isp_cus3a_api.h"
#include "mi_scl.h"
#include "mi_sensor.h"
#include "mi_sys.h"
#include "mi_vif.h"

namespace
{

constexpr MI_U16 kSocId = 0;
constexpr MI_SNR_PADID kSnrPad = 0;
constexpr MI_U32 kVifGroup = 0;
constexpr MI_U32 kVifDev = 0;
constexpr MI_U32 kIspDev = 0;
constexpr MI_U32 kIspChn = 0;
constexpr MI_U32 kIspPort = 1;
constexpr MI_U32 kSclDev = 1;
constexpr MI_U32 kSclPreviewChn = 0;
constexpr MI_U32 kSclRotateChn = 1;
constexpr MI_U32 kSclDetectChn = 5;
constexpr MI_U16 kDetectWidth = 800;
constexpr MI_U16 kDetectHeight = 480;
const char *kDefaultModel =
    "/customer/sample_code/disp_sensor2Panel_demo/model/yolov8n_800x480_P3P_fixed.sim_sgsimg.img";
constexpr MI_U32 kDispDev = 0;
constexpr MI_U32 kDispLayer = 0;
constexpr MI_U32 kDispPort = 0;
constexpr MI_U16 kSclMaxWidth = 1280;
constexpr MI_U16 kSclMaxHeight = 720;
constexpr MI_U32 kYuyvBlack = (0u << 24) | (128u << 16) | (0u << 8) | 128u;

volatile std::sig_atomic_t g_stop = 0;

void on_signal(int /*signo*/)
{
    g_stop = 1;
}

struct Options
{
    int res_index = 1;
    bool mipi = true;
    int timing = 0;
    MI_SYS_Rotate_e rotate = E_MI_SYS_ROTATE_90;
    MI_SYS_BindType_e vif_isp_bind = E_MI_SYS_BIND_TYPE_REALTIME;
    std::string iq_dir = "/customer";
    bool detect = false;
    std::string model_path = kDefaultModel;
};

struct Port
{
    MI_ModuleId_e mod = E_MI_MODULE_ID_MAX;
    MI_U32 dev = 0;
    MI_U32 chn = 0;
    MI_U32 port = 0;
};

struct Pipeline
{
    Options opt;
    MI_U32 fps = 30;
    MI_U16 panel_w = 0;
    MI_U16 panel_h = 0;

    bool sys = false;
    bool sensor = false;
    bool vif_group = false;
    bool vif_dev = false;
    bool vif_port = false;
    bool isp_device = false;
    bool isp_channel = false;
    bool isp_port = false;
    bool vif_isp_bound = false;
    bool scl_device = false;
    bool scl_preview = false;
    bool scl_rotate = false;
    bool isp_to_rotate = false;
    bool rotate_to_preview = false;
    bool isp_to_preview = false;
    bool rotate_port = false;
    bool disp_dev = false;
    bool disp_layer_bound = false;
    bool disp_layer = false;
    bool disp_port = false;
    bool scl_to_disp = false;
    bool preview_port = false;
    bool scl_detect = false;
    bool isp_to_detect = false;
    bool detect_port = false;
    bool detect_running = false;
    sensor_preview::DetectSession detect_session;
};

bool step_ok(MI_S32 ret, const char *name)
{
    if (ret != MI_SUCCESS)
    {
        std::fprintf(stderr, "%s failed, error:%x\n", name, static_cast<unsigned>(ret));
        return false;
    }
    std::printf("%s pass\n", name);
    return true;
}

void step_release(MI_S32 ret, const char *name)
{
    if (ret != MI_SUCCESS)
    {
        std::fprintf(stderr, "%s failed, error:%x\n", name, static_cast<unsigned>(ret));
        return;
    }
    std::printf("%s pass\n", name);
}

MI_SYS_ChnPort_t to_sys_port(const Port &port)
{
    MI_SYS_ChnPort_t sys_port;
    std::memset(&sys_port, 0, sizeof(sys_port));
    sys_port.eModId = port.mod;
    sys_port.u32DevId = port.dev;
    sys_port.u32ChnId = port.chn;
    sys_port.u32PortId = port.port;
    return sys_port;
}

bool bind_ports(const Port &src, const Port &dst, MI_U32 fps, MI_SYS_BindType_e bind_type)
{
    MI_SYS_ChnPort_t src_port = to_sys_port(src);
    MI_SYS_ChnPort_t dst_port = to_sys_port(dst);
    if (!step_ok(MI_SYS_BindChnPort(kSocId, &src_port, &dst_port, fps, fps, bind_type, 0),
                 "MI_SYS_BindChnPort"))
    {
        std::fprintf(stderr, "src(%d-%u-%u-%u) dst(%d-%u-%u-%u) type %d fps %u\n",
                     static_cast<int>(src.mod), src.dev, src.chn, src.port,
                     static_cast<int>(dst.mod), dst.dev, dst.chn, dst.port,
                     static_cast<int>(bind_type), fps);
        return false;
    }
    std::printf("src(%d-%u-%u-%u) dst(%d-%u-%u-%u) type %d\n",
                static_cast<int>(src.mod), src.dev, src.chn, src.port,
                static_cast<int>(dst.mod), dst.dev, dst.chn, dst.port,
                static_cast<int>(bind_type));
    return true;
}

void unbind_ports(const Port &src, const Port &dst)
{
    MI_SYS_ChnPort_t src_port = to_sys_port(src);
    MI_SYS_ChnPort_t dst_port = to_sys_port(dst);
    step_release(MI_SYS_UnBindChnPort(kSocId, &src_port, &dst_port), "MI_SYS_UnBindChnPort");
    std::printf("unbind src(%d-%u-%u-%u) dst(%d-%u-%u-%u)\n",
                static_cast<int>(src.mod), src.dev, src.chn, src.port,
                static_cast<int>(dst.mod), dst.dev, dst.chn, dst.port);
}

bool set_depth(const Port &port, MI_U32 user_depth, MI_U32 buf_depth)
{
    MI_SYS_ChnPort_t sys_port = to_sys_port(port);
    return step_ok(MI_SYS_SetChnOutputPortDepth(kSocId, &sys_port, user_depth, buf_depth),
                    "MI_SYS_SetChnOutputPortDepth");
}

MI_SYS_PixelFormat_e bayer_pixel(const MI_SNR_PlaneInfo_t &plane)
{
    if (plane.eBayerId >= E_MI_SYS_PIXEL_BAYERID_MAX)
    {
        return plane.ePixel;
    }
    return static_cast<MI_SYS_PixelFormat_e>(RGB_BAYER_PIXEL(plane.ePixPrecision, plane.eBayerId));
}

Port vif_port()
{
    return Port{E_MI_MODULE_ID_VIF, kVifDev, 0, 0};
}

Port isp_port()
{
    return Port{E_MI_MODULE_ID_ISP, kIspDev, kIspChn, kIspPort};
}

// VIF 绑到 ISP 的输入口。kIspPort(1) 是 YUV 输出口，不能当作绑定目的。
Port isp_input_port()
{
    return Port{E_MI_MODULE_ID_ISP, kIspDev, kIspChn, 0};
}

Port scl_preview_port()
{
    return Port{E_MI_MODULE_ID_SCL, kSclDev, kSclPreviewChn, 0};
}

Port scl_rotate_port()
{
    return Port{E_MI_MODULE_ID_SCL, kSclDev, kSclRotateChn, 0};
}

Port scl_detect_port()
{
    return Port{E_MI_MODULE_ID_SCL, kSclDev, kSclDetectChn, 0};
}

Port disp_port()
{
    return Port{E_MI_MODULE_ID_DISP, kDispDev, 0, 0};
}

bool rotated_90_or_270(MI_SYS_Rotate_e rotate)
{
    return rotate == E_MI_SYS_ROTATE_90 || rotate == E_MI_SYS_ROTATE_270;
}

int rotate_degree(MI_SYS_Rotate_e rotate)
{
    if (rotate == E_MI_SYS_ROTATE_90)
    {
        return 90;
    }
    if (rotate == E_MI_SYS_ROTATE_180)
    {
        return 180;
    }
    if (rotate == E_MI_SYS_ROTATE_270)
    {
        return 270;
    }
    return 0;
}

void print_usage(const char *argv0)
{
    std::printf(
        "Usage: %s [-c res] [-n 0] [-p 0|1] [-t 0|1|2] [-r 0|90|180|270] [-b 0|1] [-i iq_dir] [-d 0|1] [-m model]\n"
        "\n"
        "  -c  sensor resolution index (default 1, gc2053 1920x1080 1lane)\n"
        "  -n  must be 0 (pad0 only)\n"
        "  -p  0 TTL, 1 MIPI (default 1)\n"
        "  -t  0 USER, 1 720P60, 2 1080P60 (default 0)\n"
        "  -r  SCL rotate (default 90)\n"
        "  -b  0 VIF->ISP realtime, 1 frame (default 0)\n"
        "  -i  directory of iqApiBin0.bin (default /customer)\n"
        "  -d  0 preview only, 1 also run YOLOv8 and draw boxes (default 0)\n"
        "  -m  .img path used when -d 1\n"
        "\n"
        "Type 110 to quit.\n",
        argv0);
}

bool parse_args(int argc, char **argv, Options &opt)
{
    optind = 1;
    int flag = 0;
    while ((flag = getopt(argc, argv, "c:n:p:t:r:b:i:d:m:h")) != -1)
    {
        switch (flag)
        {
        case 'c':
            opt.res_index = std::atoi(optarg);
            break;
        case 'n':
            if (std::atoi(optarg) != 0)
            {
                std::fprintf(stderr, "only sensor pad0 (-n 0) is supported\n");
                return false;
            }
            break;
        case 'p':
            opt.mipi = std::atoi(optarg) != 0;
            std::printf("select %s panel\n", opt.mipi ? "MIPI" : "TTL");
            break;
        case 't':
            opt.timing = std::atoi(optarg);
            if (opt.timing < 0 || opt.timing > 2)
            {
                std::fprintf(stderr, "timing must be 0, 1 or 2\n");
                return false;
            }
            break;
        case 'r':
        {
            const int degree = std::atoi(optarg);
            if (degree == 0)
            {
                opt.rotate = E_MI_SYS_ROTATE_NONE;
            }
            else if (degree == 90)
            {
                opt.rotate = E_MI_SYS_ROTATE_90;
            }
            else if (degree == 180)
            {
                opt.rotate = E_MI_SYS_ROTATE_180;
            }
            else if (degree == 270)
            {
                opt.rotate = E_MI_SYS_ROTATE_270;
            }
            else
            {
                std::fprintf(stderr, "rotate must be 0, 90, 180 or 270\n");
                return false;
            }
            break;
        }
        case 'b':
            opt.vif_isp_bind = std::atoi(optarg) ? E_MI_SYS_BIND_TYPE_FRAME_BASE
                                                 : E_MI_SYS_BIND_TYPE_REALTIME;
            break;
        case 'i':
            opt.iq_dir = optarg;
            break;
        case 'd':
            opt.detect = std::atoi(optarg) != 0;
            std::printf("detect:%d\n", opt.detect ? 1 : 0);
            break;
        case 'm':
            opt.model_path = optarg;
            break;
        case 'h':
            print_usage(argv[0]);
            std::exit(0);
        default:
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

bool start_sensor(Pipeline &pipe)
{
    if (!step_ok(MI_SNR_SetPlaneMode(kSnrPad, FALSE), "MI_SNR_SetPlaneMode"))
    {
        return false;
    }
    MI_U32 res_count = 0;
    if (!step_ok(MI_SNR_QueryResCount(kSnrPad, &res_count), "MI_SNR_QueryResCount"))
    {
        return false;
    }
    for (MI_U32 i = 0; i < res_count; ++i)
    {
        MI_SNR_Res_t res;
        std::memset(&res, 0, sizeof(res));
        if (!step_ok(MI_SNR_GetRes(kSnrPad, i, &res), "MI_SNR_GetRes"))
        {
            return false;
        }
        std::printf("index %u, Crop(%u,%u,%u,%u), outputsize(%u,%u), maxfps %u, minfps %u, ResDesc %s\n",
                    i, res.stCropRect.u16X, res.stCropRect.u16Y, res.stCropRect.u16Width,
                    res.stCropRect.u16Height, res.stOutputSize.u16Width, res.stOutputSize.u16Height,
                    res.u32MaxFps, res.u32MinFps, res.strResDesc);
    }
    if (pipe.opt.res_index < 0 || static_cast<MI_U32>(pipe.opt.res_index) >= res_count)
    {
        std::fprintf(stderr, "res index %d out of range, count %u\n", pipe.opt.res_index, res_count);
        return false;
    }
    MI_SNR_Res_t selected;
    std::memset(&selected, 0, sizeof(selected));
    if (!step_ok(MI_SNR_GetRes(kSnrPad, static_cast<MI_U8>(pipe.opt.res_index), &selected),
                 "MI_SNR_GetRes") ||
        !step_ok(MI_SNR_SetRes(kSnrPad, static_cast<MI_U8>(pipe.opt.res_index)), "MI_SNR_SetRes") ||
        !step_ok(MI_SNR_Enable(kSnrPad), "MI_SNR_Enable"))
    {
        return false;
    }
    pipe.sensor = true;
    pipe.fps = selected.u32MaxFps == 0 ? 30 : selected.u32MaxFps;
    std::printf("Snr pad id:0, res count:%u, You select %d res\n", res_count, pipe.opt.res_index);
    return true;
}

bool start_vif(Pipeline &pipe)
{
    MI_SNR_PADInfo_t pad_info;
    MI_SNR_PlaneInfo_t plane;
    std::memset(&pad_info, 0, sizeof(pad_info));
    std::memset(&plane, 0, sizeof(plane));
    if (!step_ok(MI_SNR_GetPadInfo(kSnrPad, &pad_info), "MI_SNR_GetPadInfo") ||
        !step_ok(MI_SNR_GetPlaneInfo(kSnrPad, 0, &plane), "MI_SNR_GetPlaneInfo"))
    {
        return false;
    }

    MI_VIF_GroupAttr_t group_attr;
    std::memset(&group_attr, 0, sizeof(group_attr));
    group_attr.eIntfMode = E_MI_VIF_MODE_MIPI;
    group_attr.eWorkMode = E_MI_VIF_WORK_MODE_1MULTIPLEX;
    group_attr.eHDRType = E_MI_VIF_HDR_TYPE_OFF;
    group_attr.eClkEdge = E_MI_VIF_CLK_EDGE_DOUBLE;
    if (!step_ok(MI_VIF_CreateDevGroup(kVifGroup, &group_attr), "MI_VIF_CreateDevGroup"))
    {
        return false;
    }
    pipe.vif_group = true;

    MI_VIF_DevAttr_t dev_attr;
    std::memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.stInputRect.u16X = plane.stCapRect.u16X;
    dev_attr.stInputRect.u16Y = plane.stCapRect.u16Y;
    dev_attr.stInputRect.u16Width = plane.stCapRect.u16Width;
    dev_attr.stInputRect.u16Height = plane.stCapRect.u16Height;
    dev_attr.eInputPixel = bayer_pixel(plane);
    if (!step_ok(MI_VIF_SetDevAttr(kVifDev, &dev_attr), "MI_VIF_SetDevAttr") ||
        !step_ok(MI_VIF_EnableDev(kVifDev), "MI_VIF_EnableDev"))
    {
        return false;
    }
    pipe.vif_dev = true;

    MI_VIF_OutputPortAttr_t port_attr;
    std::memset(&port_attr, 0, sizeof(port_attr));
    port_attr.stCapRect = dev_attr.stInputRect;
    port_attr.stDestSize.u16Width = plane.stCapRect.u16Width;
    port_attr.stDestSize.u16Height = plane.stCapRect.u16Height;
    port_attr.eFrameRate = E_MI_VIF_FRAMERATE_FULL;
    port_attr.ePixFormat = bayer_pixel(plane);
    if (!step_ok(MI_VIF_SetOutputPortAttr(kVifDev, 0, &port_attr), "MI_VIF_SetOutputPortAttr") ||
        !set_depth(vif_port(), 0, 3) ||
        !step_ok(MI_VIF_EnableOutputPort(kVifDev, 0), "MI_VIF_EnableOutputPort"))
    {
        return false;
    }
    pipe.vif_port = true;
    std::printf("vif enabled, dev: %u, chn:0, port:0\n", kVifDev);
    return true;
}

bool start_isp(Pipeline &pipe)
{
    MI_ISP_DevAttr_t dev_attr;
    std::memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.u32DevStitchMask = E_MI_ISP_DEVICEMASK_ID0;
    if (!step_ok(MI_ISP_CreateDevice(kIspDev, &dev_attr), "MI_ISP_CreateDevice"))
    {
        return false;
    }
    pipe.isp_device = true;

    MI_ISP_ChannelAttr_t chn_attr;
    std::memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.u32SensorBindId = E_MI_ISP_SENSOR0;
    MI_ISP_ChnParam_t chn_param;
    std::memset(&chn_param, 0, sizeof(chn_param));
    chn_param.eHDRType = E_MI_ISP_HDR_TYPE_OFF;
    chn_param.e3DNRLevel = E_MI_ISP_3DNR_LEVEL1;
    chn_param.bMirror = FALSE;
    chn_param.bFlip = FALSE;
    chn_param.eRot = E_MI_SYS_ROTATE_NONE;
    if (!step_ok(MI_ISP_CreateChannel(kIspDev, kIspChn, &chn_attr), "MI_ISP_CreateChannel") ||
        !step_ok(MI_ISP_SetChnParam(kIspDev, kIspChn, &chn_param), "MI_ISP_SetChnParam") ||
        !step_ok(MI_ISP_StartChannel(kIspDev, kIspChn), "MI_ISP_StartChannel"))
    {
        return false;
    }
    pipe.isp_channel = true;

    MI_ISP_OutPortParam_t out_param;
    std::memset(&out_param, 0, sizeof(out_param));
    out_param.ePixelFormat = E_MI_SYS_PIXEL_FRAME_YUV422_YUYV;
    if (!step_ok(MI_ISP_SetOutputPortParam(kIspDev, kIspChn, kIspPort, &out_param),
                 "MI_ISP_SetOutputPortParam") ||
        !set_depth(isp_port(), 0, 3) ||
        !step_ok(MI_ISP_EnableOutputPort(kIspDev, kIspChn, kIspPort), "MI_ISP_EnableOutputPort"))
    {
        return false;
    }
    pipe.isp_port = true;

    std::string iq_path = pipe.opt.iq_dir + "/iqApiBin0.bin";
    const MI_S32 iq_ret = MI_ISP_ApiCmdLoadBinFile(kIspDev, kIspChn, iq_path.data(), 0);
    std::printf("load iq bin %s, ret:%x\n", iq_path.c_str(), static_cast<unsigned>(iq_ret));
    return true;
}

bool start_scl_channels(Pipeline &pipe)
{
    MI_SCL_DevAttr_t dev_attr;
    std::memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.u32NeedUseHWOutPortMask = E_MI_SCL_HWSCL0;
    if (!step_ok(MI_SCL_CreateDevice(kSclDev, &dev_attr), "MI_SCL_CreateDevice"))
    {
        return false;
    }
    pipe.scl_device = true;

    MI_SCL_ChannelAttr_t chn_attr;
    MI_SCL_ChnParam_t chn_param;
    std::memset(&chn_attr, 0, sizeof(chn_attr));
    std::memset(&chn_param, 0, sizeof(chn_param));
    chn_param.eRot = E_MI_SYS_ROTATE_NONE;
    if (!step_ok(MI_SCL_CreateChannel(kSclDev, kSclPreviewChn, &chn_attr), "MI_SCL_CreateChannel preview") ||
        !step_ok(MI_SCL_SetChnParam(kSclDev, kSclPreviewChn, &chn_param), "MI_SCL_SetChnParam preview") ||
        !step_ok(MI_SCL_StartChannel(kSclDev, kSclPreviewChn), "MI_SCL_StartChannel preview"))
    {
        return false;
    }
    pipe.scl_preview = true;

    if (pipe.opt.rotate == E_MI_SYS_ROTATE_NONE)
    {
        if (!bind_ports(isp_port(), scl_preview_port(), pipe.fps, E_MI_SYS_BIND_TYPE_FRAME_BASE))
        {
            return false;
        }
        pipe.isp_to_preview = true;
        return true;
    }

    chn_param.eRot = pipe.opt.rotate;
    if (!step_ok(MI_SCL_CreateChannel(kSclDev, kSclRotateChn, &chn_attr), "MI_SCL_CreateChannel rotate") ||
        !step_ok(MI_SCL_SetChnParam(kSclDev, kSclRotateChn, &chn_param), "MI_SCL_SetChnParam rotate") ||
        !step_ok(MI_SCL_StartChannel(kSclDev, kSclRotateChn), "MI_SCL_StartChannel rotate"))
    {
        return false;
    }
    pipe.scl_rotate = true;

    if (!bind_ports(isp_port(), scl_rotate_port(), pipe.fps, E_MI_SYS_BIND_TYPE_FRAME_BASE))
    {
        return false;
    }
    pipe.isp_to_rotate = true;
    if (!bind_ports(scl_rotate_port(), scl_preview_port(), pipe.fps, E_MI_SYS_BIND_TYPE_FRAME_BASE))
    {
        return false;
    }
    pipe.rotate_to_preview = true;

    MI_SCL_OutPortParam_t out_param;
    std::memset(&out_param, 0, sizeof(out_param));
    if (rotated_90_or_270(pipe.opt.rotate))
    {
        out_param.stSCLOutputSize.u16Width = 1080;
        out_param.stSCLOutputSize.u16Height = 1920;
    }
    else
    {
        out_param.stSCLOutputSize.u16Width = 1920;
        out_param.stSCLOutputSize.u16Height = 1080;
    }
    out_param.bFlip = FALSE;
    out_param.bMirror = FALSE;
    out_param.ePixelFormat = E_MI_SYS_PIXEL_FRAME_YUV422_YUYV;
    if (!step_ok(MI_SCL_SetOutputPortParam(kSclDev, kSclRotateChn, 0, &out_param),
                 "MI_SCL_SetOutputPortParam rotate") ||
        !set_depth(scl_rotate_port(), 0, 3) ||
        !step_ok(MI_SCL_EnableOutputPort(kSclDev, kSclRotateChn, 0), "MI_SCL_EnableOutputPort rotate"))
    {
        return false;
    }
    pipe.rotate_port = true;
    return true;
}

bool start_detect_scl(Pipeline &pipe)
{
    MI_SCL_ChannelAttr_t chn_attr;
    MI_SCL_ChnParam_t chn_param;
    std::memset(&chn_attr, 0, sizeof(chn_attr));
    std::memset(&chn_param, 0, sizeof(chn_param));
    chn_param.eRot = E_MI_SYS_ROTATE_NONE;
    if (!step_ok(MI_SCL_CreateChannel(kSclDev, kSclDetectChn, &chn_attr), "MI_SCL_CreateChannel detect") ||
        !step_ok(MI_SCL_SetChnParam(kSclDev, kSclDetectChn, &chn_param), "MI_SCL_SetChnParam detect") ||
        !step_ok(MI_SCL_StartChannel(kSclDev, kSclDetectChn), "MI_SCL_StartChannel detect"))
    {
        return false;
    }
    pipe.scl_detect = true;

    if (!bind_ports(isp_port(), scl_detect_port(), pipe.fps, E_MI_SYS_BIND_TYPE_FRAME_BASE))
    {
        return false;
    }
    pipe.isp_to_detect = true;

    MI_SCL_OutPortParam_t out_param;
    std::memset(&out_param, 0, sizeof(out_param));
    out_param.stSCLOutputSize.u16Width = kDetectWidth;
    out_param.stSCLOutputSize.u16Height = kDetectHeight;
    out_param.ePixelFormat = E_MI_SYS_PIXEL_FRAME_YUV_SEMIPLANAR_420;
    if (!step_ok(MI_SCL_SetOutputPortParam(kSclDev, kSclDetectChn, 0, &out_param),
                 "MI_SCL_SetOutputPortParam detect") ||
        !set_depth(scl_detect_port(), 2, 4) ||
        !step_ok(MI_SCL_EnableOutputPort(kSclDev, kSclDetectChn, 0), "MI_SCL_EnableOutputPort detect"))
    {
        return false;
    }
    pipe.detect_port = true;
    std::printf("detect scl %ux%u NV12, chn %u\n", kDetectWidth, kDetectHeight, kSclDetectChn);
    return true;
}

bool start_disp(Pipeline &pipe)
{
    MI_DISP_PubAttr_t pub_attr;
    std::memset(&pub_attr, 0, sizeof(pub_attr));
    pub_attr.u32BgColor = kYuyvBlack;
    pub_attr.eIntfType = pipe.opt.mipi ? E_MI_DISP_INTF_MIPIDSI : E_MI_DISP_INTF_TTL;
    if (pipe.opt.timing == 1)
    {
        pub_attr.eIntfSync = E_MI_DISP_OUTPUT_720P60;
    }
    else if (pipe.opt.timing == 2)
    {
        pub_attr.eIntfSync = E_MI_DISP_OUTPUT_1080P60;
    }
    else
    {
        pub_attr.eIntfSync = E_MI_DISP_OUTPUT_USER;
    }
    std::printf("panel type:%s\n", pipe.opt.mipi ? "MIPI" : "TTL");
    std::printf("disp output timing:%d\n", static_cast<int>(pub_attr.eIntfSync));
    if (!step_ok(MI_DISP_SetPubAttr(kDispDev, &pub_attr), "MI_DISP_SetPubAttr") ||
        !step_ok(MI_DISP_Enable(kDispDev), "MI_DISP_Enable"))
    {
        return false;
    }
    pipe.disp_dev = true;

    std::memset(&pub_attr, 0, sizeof(pub_attr));
    pub_attr.eIntfType = pipe.opt.mipi ? E_MI_DISP_INTF_MIPIDSI : E_MI_DISP_INTF_TTL;
    if (!step_ok(MI_DISP_GetPubAttr(kDispDev, &pub_attr), "MI_DISP_GetPubAttr"))
    {
        return false;
    }
    pipe.panel_w = pub_attr.stSyncInfo.u16Hact;
    pipe.panel_h = pub_attr.stSyncInfo.u16Vact;
    std::printf("u16Hact:%u u16Vact:%u\n", pipe.panel_w, pipe.panel_h);
    if (pipe.panel_w == 0 || pipe.panel_h == 0)
    {
        std::fprintf(stderr, "panel active size is empty\n");
        return false;
    }

    MI_DISP_VideoLayerAttr_t layer_attr;
    std::memset(&layer_attr, 0, sizeof(layer_attr));
    layer_attr.stVidLayerDispWin.u16X = 0;
    layer_attr.stVidLayerDispWin.u16Y = 0;
    layer_attr.stVidLayerDispWin.u16Width = pipe.panel_w;
    layer_attr.stVidLayerDispWin.u16Height = pipe.panel_h;
    layer_attr.stVidLayerSize.u16Width = pipe.panel_w;
    layer_attr.stVidLayerSize.u16Height = pipe.panel_h;
    if (!step_ok(MI_DISP_BindVideoLayer(kDispLayer, kDispDev), "MI_DISP_BindVideoLayer"))
    {
        return false;
    }
    pipe.disp_layer_bound = true;
    if (!step_ok(MI_DISP_SetVideoLayerAttr(kDispLayer, &layer_attr), "MI_DISP_SetVideoLayerAttr") ||
        !step_ok(MI_DISP_EnableVideoLayer(kDispLayer), "MI_DISP_EnableVideoLayer"))
    {
        return false;
    }
    pipe.disp_layer = true;

    MI_DISP_RotateConfig_t rotate_config;
    std::memset(&rotate_config, 0, sizeof(rotate_config));
    rotate_config.eRotateMode = E_MI_DISP_ROTATE_NONE;
    if (!step_ok(MI_DISP_SetVideoLayerRotateMode(kDispLayer, &rotate_config),
                 "MI_DISP_SetVideoLayerRotateMode"))
    {
        return false;
    }

    MI_DISP_InputPortAttr_t input_attr;
    std::memset(&input_attr, 0, sizeof(input_attr));
    input_attr.u16SrcWidth = pipe.panel_w;
    input_attr.u16SrcHeight = pipe.panel_h;
    input_attr.stDispWin.u16X = 0;
    input_attr.stDispWin.u16Y = 0;
    input_attr.stDispWin.u16Width = pipe.panel_w;
    input_attr.stDispWin.u16Height = pipe.panel_h;
    if (!step_ok(MI_DISP_SetInputPortAttr(kDispLayer, kDispPort, &input_attr),
                 "MI_DISP_SetInputPortAttr") ||
        !step_ok(MI_DISP_EnableInputPort(kDispLayer, kDispPort), "MI_DISP_EnableInputPort") ||
        !step_ok(MI_DISP_SetInputPortSyncMode(kDispLayer, kDispPort, E_MI_DISP_SYNC_MODE_FREE_RUN),
                 "MI_DISP_SetInputPortSyncMode"))
    {
        return false;
    }
    pipe.disp_port = true;
    return true;
}

bool enable_preview_port(Pipeline &pipe)
{
    MI_U16 out_w = pipe.panel_w;
    MI_U16 out_h = pipe.panel_h;
    if (pipe.opt.rotate == E_MI_SYS_ROTATE_NONE &&
        (out_w > kSclMaxWidth || out_h > kSclMaxHeight))
    {
        out_w = kSclMaxWidth;
        out_h = kSclMaxHeight;
    }

    MI_SCL_OutPortParam_t out_param;
    std::memset(&out_param, 0, sizeof(out_param));
    out_param.stSCLOutputSize.u16Width = out_w;
    out_param.stSCLOutputSize.u16Height = out_h;
    out_param.bFlip = FALSE;
    out_param.bMirror = FALSE;
    out_param.ePixelFormat = E_MI_SYS_PIXEL_FRAME_YUV_SEMIPLANAR_420;
    std::printf("scl output w:%u,h:%u\n", out_w, out_h);
    if (!step_ok(MI_SCL_SetOutputPortParam(kSclDev, kSclPreviewChn, 0, &out_param),
                 "MI_SCL_SetOutputPortParam preview") ||
        !set_depth(scl_preview_port(), 0, 3) ||
        !step_ok(MI_SCL_EnableOutputPort(kSclDev, kSclPreviewChn, 0),
                 "MI_SCL_EnableOutputPort preview"))
    {
        return false;
    }
    pipe.preview_port = true;
    return true;
}

bool start_pipeline(Pipeline &pipe)
{
    if (!step_ok(MI_SYS_Init(kSocId), "MI_SYS_Init"))
    {
        return false;
    }
    pipe.sys = true;
    if (!start_sensor(pipe) || !start_vif(pipe) || !start_isp(pipe))
    {
        return false;
    }
    if (!bind_ports(vif_port(), isp_input_port(), pipe.fps, pipe.opt.vif_isp_bind))
    {
        return false;
    }
    pipe.vif_isp_bound = true;
    std::printf("bind VIF(0 0 0) -> ISP(0 0), bind type:%d\n",
                static_cast<int>(pipe.opt.vif_isp_bind));
    if (!start_scl_channels(pipe) || (pipe.opt.detect && !start_detect_scl(pipe)) || !start_disp(pipe))
    {
        return false;
    }
    if (!bind_ports(scl_preview_port(), disp_port(), pipe.fps, E_MI_SYS_BIND_TYPE_FRAME_BASE))
    {
        return false;
    }
    pipe.scl_to_disp = true;
    if (!enable_preview_port(pipe))
    {
        return false;
    }
    if (!pipe.opt.detect)
    {
        return true;
    }
    sensor_preview::DetectConfig detect_config;
    detect_config.model_path = pipe.opt.model_path;
    detect_config.rotate_degree = rotate_degree(pipe.opt.rotate);
    if (!pipe.detect_session.start(detect_config))
    {
        return false;
    }
    pipe.detect_running = true;
    return true;
}

void stop_pipeline(Pipeline &pipe)
{
    if (pipe.detect_running)
    {
        pipe.detect_session.stop();
        pipe.detect_running = false;
    }
    if (pipe.detect_port)
    {
        step_release(MI_SCL_DisableOutputPort(kSclDev, kSclDetectChn, 0),
                     "MI_SCL_DisableOutputPort detect");
    }
    if (pipe.isp_to_detect)
    {
        unbind_ports(isp_port(), scl_detect_port());
    }
    if (pipe.scl_detect)
    {
        step_release(MI_SCL_StopChannel(kSclDev, kSclDetectChn), "MI_SCL_StopChannel detect");
        step_release(MI_SCL_DestroyChannel(kSclDev, kSclDetectChn), "MI_SCL_DestroyChannel detect");
    }
    if (pipe.scl_to_disp)
    {
        unbind_ports(scl_preview_port(), disp_port());
    }
    if (pipe.preview_port)
    {
        step_release(MI_SCL_DisableOutputPort(kSclDev, kSclPreviewChn, 0),
                     "MI_SCL_DisableOutputPort preview");
    }
    if (pipe.disp_port)
    {
        step_release(MI_DISP_DisableInputPort(kDispLayer, kDispPort), "MI_DISP_DisableInputPort");
    }
    if (pipe.disp_layer)
    {
        step_release(MI_DISP_DisableVideoLayer(kDispLayer), "MI_DISP_DisableVideoLayer");
    }
    if (pipe.disp_layer_bound)
    {
        step_release(MI_DISP_UnBindVideoLayer(kDispLayer, kDispDev), "MI_DISP_UnBindVideoLayer");
    }
    if (pipe.disp_dev)
    {
        step_release(MI_DISP_Disable(kDispDev), "MI_DISP_Disable");
    }

    if (pipe.rotate_port)
    {
        step_release(MI_SCL_DisableOutputPort(kSclDev, kSclRotateChn, 0),
                     "MI_SCL_DisableOutputPort rotate");
    }
    if (pipe.rotate_to_preview)
    {
        unbind_ports(scl_rotate_port(), scl_preview_port());
    }
    if (pipe.isp_to_rotate)
    {
        unbind_ports(isp_port(), scl_rotate_port());
    }
    if (pipe.scl_rotate)
    {
        step_release(MI_SCL_StopChannel(kSclDev, kSclRotateChn), "MI_SCL_StopChannel rotate");
        step_release(MI_SCL_DestroyChannel(kSclDev, kSclRotateChn), "MI_SCL_DestroyChannel rotate");
    }
    if (pipe.isp_to_preview)
    {
        unbind_ports(isp_port(), scl_preview_port());
    }
    if (pipe.scl_preview)
    {
        step_release(MI_SCL_StopChannel(kSclDev, kSclPreviewChn), "MI_SCL_StopChannel preview");
        step_release(MI_SCL_DestroyChannel(kSclDev, kSclPreviewChn), "MI_SCL_DestroyChannel preview");
    }
    if (pipe.scl_device)
    {
        step_release(MI_SCL_DestroyDevice(kSclDev), "MI_SCL_DestroyDevice");
    }

    if (pipe.isp_port)
    {
        step_release(MI_ISP_DisableOutputPort(kIspDev, kIspChn, kIspPort), "MI_ISP_DisableOutputPort");
    }
    if (pipe.vif_isp_bound)
    {
        unbind_ports(vif_port(), isp_input_port());
    }
    if (pipe.isp_channel)
    {
        step_release(MI_ISP_StopChannel(kIspDev, kIspChn), "MI_ISP_StopChannel");
        step_release(MI_ISP_DestroyChannel(kIspDev, kIspChn), "MI_ISP_DestroyChannel");
    }
    if (pipe.isp_device)
    {
        step_release(MI_ISP_DestoryDevice(kIspDev), "MI_ISP_DestoryDevice");
    }

    if (pipe.vif_port)
    {
        step_release(MI_VIF_DisableOutputPort(kVifDev, 0), "MI_VIF_DisableOutputPort");
    }
    if (pipe.vif_dev)
    {
        step_release(MI_VIF_DisableDev(kVifDev), "MI_VIF_DisableDev");
    }
    if (pipe.vif_group)
    {
        step_release(MI_VIF_DestroyDevGroup(kVifGroup), "MI_VIF_DestroyDevGroup");
    }
    if (pipe.sensor)
    {
        step_release(MI_SNR_Disable(kSnrPad), "MI_SNR_Disable");
    }
    if (pipe.sys)
    {
        step_release(MI_SYS_Exit(kSocId), "MI_SYS_Exit");
    }
}

}  // namespace

int main(int argc, char **argv)
{
    Options opt;
    if (!parse_args(argc, argv, opt))
    {
        return 1;
    }

    struct sigaction action;
    std::memset(&action, 0, sizeof(action));
    action.sa_handler = on_signal;
    sigaction(SIGINT, &action, nullptr);

    Pipeline pipe;
    pipe.opt = opt;
    if (!start_pipeline(pipe))
    {
        stop_pipeline(pipe);
        return 1;
    }

    std::printf("select 110: quit app\n");
    while (g_stop == 0)
    {
        int choice = -1;
        if (std::scanf("%d", &choice) != 1)
        {
            break;
        }
        if (choice == 110)
        {
            std::printf("quit app\n");
            break;
        }
        std::printf("select 110: quit app\n");
    }

    stop_pipeline(pipe);
    return 0;
}
