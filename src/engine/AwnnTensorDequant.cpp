#ifdef AWNN_FOUND

#include "deploy_percept/engine/AwnnTensorDequant.hpp"

#include "deploy_percept/post_process/YoloBasePostProcess.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace deploy_percept
{
    namespace engine
    {

        namespace
        {

            using deploy_percept::post_process::YoloBasePostProcess;

            float int8ToFp32(int8_t val, int8_t fixed_point_pos)
            {
                if (fixed_point_pos > 0)
                {
                    return static_cast<float>(val) * (1.f / static_cast<float>(1 << fixed_point_pos));
                }
                return static_cast<float>(val) * static_cast<float>(1 << -fixed_point_pos);
            }

            float int16ToFp32(int16_t val, int8_t fixed_point_pos)
            {
                if (fixed_point_pos > 0)
                {
                    return static_cast<float>(val) * (1.f / static_cast<float>(1 << fixed_point_pos));
                }
                return static_cast<float>(val) * static_cast<float>(1 << -fixed_point_pos);
            }

            float uint8ToFp32(uint8_t val, int32_t zero_point, float scale)
            {
                return (static_cast<float>(val) - static_cast<float>(zero_point)) * scale;
            }

            union Fp32Bits
            {
                uint32_t u;
                float f;
            };

            float fp16ToFp32(int16_t in)
            {
                const Fp32Bits magic = {static_cast<uint32_t>((254 - 15) << 23)};
                const Fp32Bits infnan = {static_cast<uint32_t>((127 + 16) << 23)};
                Fp32Bits o{};
                o.u = (static_cast<uint16_t>(in) & 0x7FFFu) << 13u;
                o.f *= magic.f;
                if (o.f >= infnan.f)
                {
                    o.u |= 255u << 23u;
                }
                o.u |= (static_cast<uint16_t>(in) & 0x8000u) << 16u;
                return o.f;
            }

        } // namespace

        void AwnnTensorDequant::buildQuantLut(TensorDesc &desc)
        {
            desc.has_quant_lut = false;
            if (desc.data_format == VIP_BUFFER_FORMAT_UINT8)
            {
                for (int j = 0; j < 256; ++j)
                {
                    desc.quant_lut[static_cast<size_t>(j)] =
                        uint8ToFp32(static_cast<uint8_t>(j), desc.zero_point, desc.scale);
                }
                desc.has_quant_lut = true;
            }
            else if (desc.data_format == VIP_BUFFER_FORMAT_INT8)
            {
                if (desc.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
                {
                    for (int j = 0; j < 256; ++j)
                    {
                        const auto q = static_cast<int8_t>(static_cast<uint8_t>(j));
                        desc.quant_lut[static_cast<size_t>(j)] =
                            int8ToFp32(q, static_cast<int8_t>(desc.fixed_point_pos));
                    }
                    desc.has_quant_lut = true;
                }
                else if (desc.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM)
                {
                    for (int j = 0; j < 256; ++j)
                    {
                        const auto q = static_cast<int8_t>(static_cast<uint8_t>(j));
                        desc.quant_lut[static_cast<size_t>(j)] =
                            YoloBasePostProcess::deqntAffineToF32(q, desc.zero_point, desc.scale);
                    }
                    desc.has_quant_lut = true;
                }
            }
        }

        bool AwnnTensorDequant::toFloat(
            const TensorDesc &desc,
            const void *raw_data,
            std::size_t raw_byte_size,
            std::vector<float> &out_float)
        {
            if (raw_data == nullptr || desc.element_count == 0)
            {
                return false;
            }

            out_float.resize(desc.element_count);

            if (desc.data_format == VIP_BUFFER_FORMAT_FP32)
            {
                if (raw_byte_size < desc.element_count * sizeof(float))
                {
                    return false;
                }
                std::memcpy(out_float.data(), raw_data, desc.element_count * sizeof(float));
                return true;
            }

            if (desc.data_format == VIP_BUFFER_FORMAT_FP16)
            {
                if (raw_byte_size < desc.element_count * sizeof(uint16_t))
                {
                    return false;
                }
                const auto *src = static_cast<const int16_t *>(raw_data);
                for (vip_uint32_t i = 0; i < desc.element_count; ++i)
                {
                    out_float[i] = fp16ToFp32(src[i]);
                }
                return true;
            }

            if ((desc.data_format == VIP_BUFFER_FORMAT_UINT8 || desc.data_format == VIP_BUFFER_FORMAT_INT8) &&
                desc.has_quant_lut)
            {
                if (raw_byte_size < desc.element_count)
                {
                    return false;
                }
                const auto *src = static_cast<const uint8_t *>(raw_data);
                for (vip_uint32_t i = 0; i < desc.element_count; ++i)
                {
                    out_float[i] = desc.quant_lut[src[i]];
                }
                return true;
            }

            if (desc.data_format == VIP_BUFFER_FORMAT_INT16)
            {
                if (raw_byte_size < desc.element_count * sizeof(int16_t))
                {
                    return false;
                }
                const auto *src = static_cast<const int16_t *>(raw_data);
                for (vip_uint32_t i = 0; i < desc.element_count; ++i)
                {
                    out_float[i] = int16ToFp32(src[i], static_cast<int8_t>(desc.fixed_point_pos));
                }
                return true;
            }

            std::fprintf(
                stderr,
                "AwnnTensorDequant: unsupported output format=%d quant=%d\n",
                desc.data_format,
                desc.quant_format);
            return false;
        }

    } // namespace engine
} // namespace deploy_percept

#endif
