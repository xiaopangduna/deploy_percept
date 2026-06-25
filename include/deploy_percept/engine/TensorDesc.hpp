#pragma once

#ifdef AWNN_FOUND

extern "C" {
#include "vip_lite.h"
}

#include <array>
#include <cstdint>
#include <string>

namespace deploy_percept
{
    namespace engine
    {

        /** VIPLite tensor 元数据（含可选 INT8/UINT8 反量化查找表） */
        struct TensorDesc
        {
            vip_enum data_format{VIP_BUFFER_FORMAT_FP32};
            vip_enum quant_format{VIP_BUFFER_QUANTIZE_NONE};
            std::array<vip_uint32_t, 6> dims{};
            vip_uint32_t num_dims{0};
            vip_uint32_t element_count{0};
            vip_uint32_t byte_size{0};

            vip_float_t scale{1.f};
            vip_int32_t zero_point{0};
            vip_int32_t fixed_point_pos{0};
            std::string name;

            std::array<float, 256> quant_lut{};
            bool has_quant_lut{false};
        };

    } // namespace engine
} // namespace deploy_percept

#endif
