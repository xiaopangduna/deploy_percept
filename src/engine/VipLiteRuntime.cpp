#ifdef AWNN_FOUND

#include "deploy_percept/engine/VipLiteRuntime.hpp"

extern "C" {
#include "vip_lite.h"
}

#include <cstdio>

namespace deploy_percept
{
    namespace engine
    {

        int VipLiteRuntime::ref_count_ = 0;

        VipLiteRuntime::VipLiteRuntime()
        {
            if (ref_count_++ == 0)
            {
                const vip_status_e status = vip_init();
                if (status != VIP_SUCCESS)
                {
                    std::fprintf(stderr, "VipLiteRuntime: vip_init failed: %d\n", status);
                    --ref_count_;
                    return;
                }
            }
            acquired_ = true;
            ok_ = true;
        }

        VipLiteRuntime::~VipLiteRuntime()
        {
            if (!acquired_)
            {
                return;
            }
            acquired_ = false;
            ok_ = false;
            if (ref_count_ > 0 && --ref_count_ == 0)
            {
                const vip_status_e status = vip_destroy();
                if (status != VIP_SUCCESS)
                {
                    std::fprintf(stderr, "VipLiteRuntime: vip_destroy failed: %d\n", status);
                }
            }
        }

    } // namespace engine
} // namespace deploy_percept

#endif
