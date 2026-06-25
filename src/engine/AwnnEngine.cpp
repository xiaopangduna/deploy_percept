#ifdef AWNN_FOUND

#include "deploy_percept/engine/AwnnEngine.hpp"

extern "C" {
#include "vip_lite.h"
}

#include <cstdio>
#include <cstring>
#include <chrono>

namespace deploy_percept
{
    namespace engine
    {

        namespace
        {

            struct IoTensor
            {
                vip_buffer buffer{nullptr};
                vip_buffer_create_params_t create_param{};
                vip_uint32_t byte_size{0};
                float scale{1.f};
                int32_t zero_point{0};
            };

            bool fillCreateParam(
                vip_network network,
                vip_uint32_t index,
                bool is_output,
                IoTensor &io)
            {
                std::memset(&io.create_param, 0, sizeof(io.create_param));
                io.create_param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;

                const auto query = is_output ? vip_query_output : vip_query_input;
                if (query(network, index, VIP_BUFFER_PROP_DATA_FORMAT, &io.create_param.data_format) !=
                    VIP_SUCCESS)
                {
                    return false;
                }
                if (query(network, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &io.create_param.num_of_dims) !=
                    VIP_SUCCESS)
                {
                    return false;
                }
                if (query(
                        network, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, io.create_param.sizes) !=
                    VIP_SUCCESS)
                {
                    return false;
                }
                if (query(network, index, VIP_BUFFER_PROP_QUANT_FORMAT, &io.create_param.quant_format) !=
                    VIP_SUCCESS)
                {
                    return false;
                }

                switch (io.create_param.quant_format)
                {
                case VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:
                    query(
                        network,
                        index,
                        VIP_BUFFER_PROP_FIXED_POINT_POS,
                        &io.create_param.quant_data.dfp.fixed_point_pos);
                    break;
                case VIP_BUFFER_QUANTIZE_TF_ASYMM:
                    query(
                        network,
                        index,
                        VIP_BUFFER_PROP_TF_SCALE,
                        &io.create_param.quant_data.affine.scale);
                    query(
                        network,
                        index,
                        VIP_BUFFER_PROP_TF_ZERO_POINT,
                        &io.create_param.quant_data.affine.zeroPoint);
                    io.scale = io.create_param.quant_data.affine.scale;
                    io.zero_point = io.create_param.quant_data.affine.zeroPoint;
                    break;
                default:
                    break;
                }

                return true;
            }

            bool isInt8Format(int format)
            {
                return format == VIP_BUFFER_FORMAT_INT8 || format == VIP_BUFFER_FORMAT_UINT8;
            }

            bool isTypicalChannelCount(vip_uint32_t c)
            {
                return c == 1 || c == 3 || c == 4;
            }

            /** 从 VIPLite sizes 解析 C/H/W，兼容 NCHW / NHWC，并用 byte_size 校验 */
            bool parseInputImageDims(
                vip_uint32_t num_dims,
                const vip_uint32_t *sizes,
                vip_uint32_t byte_size,
                std::uint32_t &channels,
                std::uint32_t &height,
                std::uint32_t &width)
            {
                struct Candidate
                {
                    vip_uint32_t c;
                    vip_uint32_t h;
                    vip_uint32_t w;
                };

                Candidate candidates[4]{};
                std::size_t candidate_count = 0;

                if (num_dims == 4)
                {
                    // [N,C,H,W]、[N,H,W,C]、[H,W,C,N]、[C,H,W,N] 等 VIPLite 常见报告方式
                    candidates[0] = {sizes[1], sizes[2], sizes[3]}; // NCHW
                    candidates[1] = {sizes[3], sizes[1], sizes[2]}; // NHWC
                    candidates[2] = {sizes[2], sizes[0], sizes[1]}; // HWC + padding，如 [640,640,3,1]
                    candidates[3] = {sizes[0], sizes[1], sizes[2]}; // CHW + padding，如 [3,640,640,1]
                    candidate_count = 4;
                }
                else if (num_dims == 3)
                {
                    candidates[0] = {sizes[0], sizes[1], sizes[2]}; // CHW
                    candidates[1] = {sizes[2], sizes[0], sizes[1]}; // HWC
                    candidate_count = 2;
                }
                else
                {
                    return false;
                }

                const Candidate *best = nullptr;
                const Candidate *fallback = nullptr;

                for (std::size_t i = 0; i < candidate_count; ++i)
                {
                    const Candidate &cand = candidates[i];
                    if (cand.c == 0 || cand.h == 0 || cand.w == 0)
                    {
                        continue;
                    }
                    const std::uint64_t product =
                        static_cast<std::uint64_t>(cand.c) * cand.h * cand.w;
                    if (product != byte_size)
                    {
                        continue;
                    }
                    if (isTypicalChannelCount(cand.c))
                    {
                        best = &cand;
                        break;
                    }
                    if (fallback == nullptr)
                    {
                        fallback = &cand;
                    }
                }

                const Candidate *picked = best != nullptr ? best : fallback;
                if (picked == nullptr)
                {
                    std::fprintf(
                        stderr,
                        "AwnnEngine: cannot parse input shape (num_dims=%u byte_size=%u",
                        num_dims,
                        byte_size);
                    for (vip_uint32_t d = 0; d < num_dims; ++d)
                    {
                        std::fprintf(stderr, " sizes[%u]=%u", d, sizes[d]);
                    }
                    std::fprintf(stderr, ")\n");
                    return false;
                }

                channels = picked->c;
                height = picked->h;
                width = picked->w;
                return true;
            }

        } // namespace

        int AwnnEngine::runtime_ref_count_ = 0;

        void AwnnEngine::acquireRuntime()
        {
            if (runtime_ref_count_++ == 0)
            {
                const vip_status_e status = vip_init();
                if (status != VIP_SUCCESS)
                {
                    std::fprintf(stderr, "AwnnEngine: vip_init failed: %d\n", status);
                    --runtime_ref_count_;
                    return;
                }
            }
            runtime_acquired_ = true;
        }

        void AwnnEngine::releaseRuntime()
        {
            if (!runtime_acquired_)
            {
                return;
            }
            runtime_acquired_ = false;
            if (runtime_ref_count_ > 0 && --runtime_ref_count_ == 0)
            {
                const vip_status_e status = vip_destroy();
                if (status != VIP_SUCCESS)
                {
                    std::fprintf(stderr, "AwnnEngine: vip_destroy failed: %d\n", status);
                }
            }
        }

        AwnnEngine::AwnnEngine(const Params &params)
            : params_(params)
        {
            acquireRuntime();
            if (!runtime_acquired_)
            {
                return;
            }

            valid_ = createNetwork();
            if (!valid_)
            {
                destroyNetwork();
            }
        }

        AwnnEngine::~AwnnEngine()
        {
            destroyNetwork();
        }

        bool AwnnEngine::createNetwork()
        {
            vip_network network = nullptr;
            vip_status_e status = vip_create_network(
                params_.model_path.c_str(),
                0,
                VIP_CREATE_NETWORK_FROM_FILE,
                &network);
            if (status != VIP_SUCCESS || network == nullptr)
            {
                std::fprintf(
                    stderr,
                    "AwnnEngine: vip_create_network failed (%d): %s\n",
                    status,
                    params_.model_path.c_str());
                return false;
            }
            network_ = network;

            if (!prepareIo())
            {
                return false;
            }

            status = vip_prepare_network(network);
            if (status != VIP_SUCCESS)
            {
                std::fprintf(stderr, "AwnnEngine: vip_prepare_network failed: %d\n", status);
                return false;
            }
            network_prepared_ = true;

            for (vip_uint32_t i = 0; i < input_buffers_.size(); ++i)
            {
                status = vip_set_input(network, i, static_cast<vip_buffer>(input_buffers_[i]));
                if (status != VIP_SUCCESS)
                {
                    std::fprintf(stderr, "AwnnEngine: vip_set_input(%u) failed: %d\n", i, status);
                    return false;
                }
            }

            for (vip_uint32_t i = 0; i < output_buffers_vip_.size(); ++i)
            {
                status = vip_set_output(network, i, static_cast<vip_buffer>(output_buffers_vip_[i]));
                if (status != VIP_SUCCESS)
                {
                    std::fprintf(stderr, "AwnnEngine: vip_set_output(%u) failed: %d\n", i, status);
                    return false;
                }
            }

            output_raw_storage_.resize(output_buffers_vip_.size());
            output_mapped_.assign(output_buffers_vip_.size(), false);
            output_ptrs_.resize(output_buffers_vip_.size(), nullptr);
            output_float_ptrs_.resize(output_buffers_vip_.size(), nullptr);
            return true;
        }

        bool AwnnEngine::prepareIo()
        {
            auto network = static_cast<vip_network>(network_);
            vip_uint32_t input_count = 0;
            vip_uint32_t output_count = 0;

            if (vip_query_network(network, VIP_NETWORK_PROP_INPUT_COUNT, &input_count) != VIP_SUCCESS)
            {
                return false;
            }
            if (vip_query_network(network, VIP_NETWORK_PROP_OUTPUT_COUNT, &output_count) != VIP_SUCCESS)
            {
                return false;
            }

            input_buffers_.clear();
            input_byte_sizes_.clear();
            input_attrs_.clear();
            output_buffers_vip_.clear();
            output_byte_sizes_.clear();
            output_formats_.clear();
            output_scales_.clear();
            output_zps_.clear();
            outputs_are_int8_ = true;
            outputs_are_fp32_ = true;

            for (vip_uint32_t i = 0; i < input_count; ++i)
            {
                IoTensor io{};
                if (!fillCreateParam(network, i, false, io))
                {
                    return false;
                }

                vip_buffer buffer = nullptr;
                if (vip_create_buffer(&io.create_param, 0, &buffer) != VIP_SUCCESS || buffer == nullptr)
                {
                    std::fprintf(stderr, "AwnnEngine: create input buffer %u failed\n", i);
                    return false;
                }
                io.byte_size = vip_get_buffer_size(buffer);

                InputAttr attr{};
                attr.dims.assign(
                    io.create_param.sizes,
                    io.create_param.sizes + io.create_param.num_of_dims);
                if (!parseInputImageDims(
                        io.create_param.num_of_dims,
                        io.create_param.sizes,
                        io.byte_size,
                        attr.channels,
                        attr.height,
                        attr.width))
                {
                    std::fprintf(
                        stderr,
                        "AwnnEngine: unsupported input shape (num_dims=%u)\n",
                        io.create_param.num_of_dims);
                    vip_destroy_buffer(buffer);
                    return false;
                }

                input_buffers_.push_back(buffer);
                input_byte_sizes_.push_back(io.byte_size);
                input_attrs_.push_back(attr);
            }

            for (vip_uint32_t i = 0; i < output_count; ++i)
            {
                IoTensor io{};
                if (!fillCreateParam(network, i, true, io))
                {
                    return false;
                }

                vip_buffer buffer = nullptr;
                if (vip_create_buffer(&io.create_param, 0, &buffer) != VIP_SUCCESS || buffer == nullptr)
                {
                    std::fprintf(stderr, "AwnnEngine: create output buffer %u failed\n", i);
                    return false;
                }
                io.byte_size = vip_get_buffer_size(buffer);
                if (isInt8Format(io.create_param.data_format))
                {
                    outputs_are_fp32_ = false;
                }
                else if (io.create_param.data_format == VIP_BUFFER_FORMAT_FP32)
                {
                    outputs_are_int8_ = false;
                }
                else
                {
                    outputs_are_int8_ = false;
                    outputs_are_fp32_ = false;
                }
                output_buffers_vip_.push_back(buffer);
                output_byte_sizes_.push_back(io.byte_size);
                output_formats_.push_back(static_cast<int>(io.create_param.data_format));
                output_scales_.push_back(io.scale);
                output_zps_.push_back(io.zero_point);
            }

            return true;
        }

        void AwnnEngine::destroyNetwork()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                release_outputs_unlocked();
            }

            if (network_ != nullptr)
            {
                auto network = static_cast<vip_network>(network_);
                if (network_prepared_)
                {
                    vip_finish_network(network);
                    network_prepared_ = false;
                }

                for (void *buf : input_buffers_)
                {
                    if (buf != nullptr)
                    {
                        vip_destroy_buffer(static_cast<vip_buffer>(buf));
                    }
                }
                for (void *buf : output_buffers_vip_)
                {
                    if (buf != nullptr)
                    {
                        vip_destroy_buffer(static_cast<vip_buffer>(buf));
                    }
                }

                vip_destroy_network(network);
                network_ = nullptr;
            }

            input_buffers_.clear();
            input_byte_sizes_.clear();
            input_attrs_.clear();
            output_buffers_vip_.clear();
            output_byte_sizes_.clear();
            output_formats_.clear();
            output_raw_storage_.clear();
            output_mapped_.clear();
            output_ptrs_.clear();
            output_float_ptrs_.clear();
            output_scales_.clear();
            output_zps_.clear();
            outputs_are_int8_ = false;
            outputs_are_fp32_ = false;
            outputs_ready_ = false;
            output_storage_ = OutputStorage::None;
            last_run_timing_ = {};
            valid_ = false;

            releaseRuntime();
        }

        bool AwnnEngine::run(const void *input_buffer, std::size_t input_byte_size)
        {
            const bool copy_to_host = params_.output_fetch == OutputFetch::HostCopy;
            return run_impl(input_buffer, input_byte_size, copy_to_host);
        }

        void AwnnEngine::release_outputs()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (output_storage_ != OutputStorage::Mapped)
            {
                return;
            }
            release_outputs_unlocked();
        }

        bool AwnnEngine::run_impl(const void *input_buffer, std::size_t input_byte_size, bool copy_outputs_to_host)
        {
            if (!valid_ || input_buffer == nullptr || input_buffers_.empty())
            {
                return false;
            }

            std::lock_guard<std::mutex> lock(mutex_);
            release_outputs_unlocked();

            if (!copyInput(input_buffer, input_byte_size))
            {
                return false;
            }

            auto network = static_cast<vip_network>(network_);
            const auto t0 = std::chrono::steady_clock::now();
            if (vip_run_network(network) != VIP_SUCCESS)
            {
                std::fprintf(stderr, "AwnnEngine: vip_run_network failed\n");
                return false;
            }
            const auto t1 = std::chrono::steady_clock::now();

            const bool fetched =
                copy_outputs_to_host ? fetch_outputs_copy() : fetch_outputs_mapped();
            const auto t2 = std::chrono::steady_clock::now();
            if (!fetched)
            {
                release_outputs_unlocked();
                return false;
            }

            output_storage_ = copy_outputs_to_host ? OutputStorage::HostCopy : OutputStorage::Mapped;
            outputs_ready_ = true;
            last_run_timing_.npu_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            last_run_timing_.output_fetch_ms =
                std::chrono::duration<double, std::milli>(t2 - t1).count();
            return true;
        }

        void AwnnEngine::release_outputs_unlocked()
        {
            for (std::uint32_t i = 0; i < output_buffers_vip_.size(); ++i)
            {
                if (i < output_mapped_.size() && output_mapped_[i])
                {
                    vip_unmap_buffer(static_cast<vip_buffer>(output_buffers_vip_[i]));
                    output_mapped_[i] = false;
                }
                if (i < output_ptrs_.size())
                {
                    output_ptrs_[i] = nullptr;
                }
                if (i < output_float_ptrs_.size())
                {
                    output_float_ptrs_[i] = nullptr;
                }
            }

            outputs_ready_ = false;
            output_storage_ = OutputStorage::None;
        }

        bool AwnnEngine::fetch_outputs_mapped()
        {
            for (std::uint32_t i = 0; i < output_buffers_vip_.size(); ++i)
            {
                auto *buffer = static_cast<vip_buffer>(output_buffers_vip_[i]);
                if (vip_flush_buffer(buffer, VIP_BUFFER_OPER_TYPE_INVALIDATE) != VIP_SUCCESS)
                {
                    release_outputs_unlocked();
                    return false;
                }

                void *out_data = vip_map_buffer(buffer);
                if (out_data == nullptr)
                {
                    release_outputs_unlocked();
                    return false;
                }

                output_ptrs_[i] = reinterpret_cast<int8_t *>(out_data);
                output_float_ptrs_[i] = reinterpret_cast<float *>(out_data);
                output_mapped_[i] = true;
            }

            return true;
        }

        bool AwnnEngine::fetch_outputs_copy()
        {
            for (std::uint32_t i = 0; i < output_buffers_vip_.size(); ++i)
            {
                auto *buffer = static_cast<vip_buffer>(output_buffers_vip_[i]);
                if (vip_flush_buffer(buffer, VIP_BUFFER_OPER_TYPE_INVALIDATE) != VIP_SUCCESS)
                {
                    return false;
                }

                void *out_data = vip_map_buffer(buffer);
                if (out_data == nullptr)
                {
                    return false;
                }

                auto &raw = output_raw_storage_[i];
                raw.resize(output_byte_sizes_[i]);
                std::memcpy(raw.data(), out_data, output_byte_sizes_[i]);
                output_ptrs_[i] = reinterpret_cast<int8_t *>(raw.data());
                output_float_ptrs_[i] = reinterpret_cast<float *>(raw.data());
                output_mapped_[i] = false;
                vip_unmap_buffer(buffer);
            }

            return true;
        }

        bool AwnnEngine::copyInput(const void *user_input, std::size_t input_byte_size)
        {
            auto *buffer = static_cast<vip_buffer>(input_buffers_[0]);
            const vip_uint32_t buf_size = vip_get_buffer_size(buffer);
            if (input_byte_size < buf_size)
            {
                std::fprintf(
                    stderr,
                    "AwnnEngine: input buffer too small: got %zu bytes, model expects %u\n",
                    input_byte_size,
                    buf_size);
                return false;
            }

            void *device_data = vip_map_buffer(buffer);
            if (device_data == nullptr)
            {
                std::fprintf(stderr, "AwnnEngine: vip_map_buffer(input) failed\n");
                return false;
            }

            std::memcpy(device_data, user_input, buf_size);
            vip_unmap_buffer(buffer);

            if (vip_flush_buffer(buffer, VIP_BUFFER_OPER_TYPE_FLUSH) != VIP_SUCCESS)
            {
                std::fprintf(stderr, "AwnnEngine: vip_flush_buffer(input) failed\n");
                return false;
            }

            return true;
        }

        int8_t **AwnnEngine::output_buffers()
        {
            if (!valid_ || !outputs_ready_ || !outputs_are_int8_ || output_ptrs_.empty())
            {
                return nullptr;
            }
            return output_ptrs_.data();
        }

        float **AwnnEngine::output_buffers_float()
        {
            if (!valid_ || !outputs_ready_ || !outputs_are_fp32_ || output_float_ptrs_.empty())
            {
                return nullptr;
            }
            return output_float_ptrs_.data();
        }

        float *AwnnEngine::output_float(std::uint32_t index)
        {
            if (!valid_ || !outputs_ready_ || !outputs_are_fp32_ || index >= output_float_ptrs_.size())
            {
                return nullptr;
            }
            return output_float_ptrs_[index];
        }

        int8_t *AwnnEngine::output_int8(std::uint32_t index)
        {
            if (!valid_ || !outputs_ready_ || !outputs_are_int8_ || index >= output_ptrs_.size())
            {
                return nullptr;
            }
            return output_ptrs_[index];
        }

    } // namespace engine
} // namespace deploy_percept

#endif
