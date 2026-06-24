/**
 * Allwinner VIPLite 链接验证（暂不调用 NPU 推理）。
 * 见 cmake/modules/FindAwnn.cmake
 */
#include <cstdio>

extern "C" {
#include "vip_lite.h"
}

int main()
{
    auto init_fn = &vip_init;
    auto destroy_fn = &vip_destroy;

    std::printf("yolov5_detect_awnn link smoke\n");
    std::printf("  vip_init    = %p\n", reinterpret_cast<void *>(init_fn));
    std::printf("  vip_destroy = %p\n", reinterpret_cast<void *>(destroy_fn));

    return 0;
}
