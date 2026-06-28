#ifdef AWNN_FOUND

#include "deploy_percept/engine/AwnnEngine.hpp"

extern "C" {
#include "vip_lite.h"
}

#include <cstdio>
#include <cstring>
#include <optional>

namespace deploy_percept
{
    namespace engine
    {

        namespace
        {

            bool queryCreateParam(
                vip_network network,
                vip_uint32_t index,
                bool is_output,
                vip_buffer_create_params_t &param)
            {
                std::memset(&param, 0, sizeof(param));
                param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;

                const auto query = is_output ? vip_query_output : vip_query_input;
                if (query(network, index, VIP_BUFFER_PROP_DATA_FORMAT, &param.data_format) != VIP_SUCCESS)
                {
                    return false;
                }
                if (query(network, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &param.num_of_dims) != VIP_SUCCESS)
                {
                    return false;
                }
                if (query(network, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, param.sizes) != VIP_SUCCESS)
                {
                    return false;
                }
                if (query(network, index, VIP_BUFFER_PROP_QUANT_FORMAT, &param.quant_format) != VIP_SUCCESS)
                {
                    return false;
                }

                if (param.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
                {
                    query(
                        network,
                        index,
                        VIP_BUFFER_PROP_FIXED_POINT_POS,
                        &param.quant_data.dfp.fixed_point_pos);
                }
                else if (param.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM)
                {
                    query(network, index, VIP_BUFFER_PROP_TF_SCALE, &param.quant_data.affine.scale);
                    query(
                        network,
                        index,
                        VIP_BUFFER_PROP_TF_ZERO_POINT,
                        &param.quant_data.affine.zeroPoint);
                }

                return true;
            }

            void logPossibleInputShapes(
                vip_uint32_t num_dims,
                const vip_uint32_t *sizes,
                vip_uint32_t byte_size)
            {
                struct Candidate
                {
                    const char *label;
                    vip_uint32_t c;
                    vip_uint32_t h;
                    vip_uint32_t w;
                };

                Candidate candidates[4]{};
                std::size_t candidate_count = 0;

                if (num_dims == 4)
                {
                    candidates[0] = {"NCHW", sizes[1], sizes[2], sizes[3]};
                    candidates[1] = {"NHWC", sizes[3], sizes[1], sizes[2]};
                    candidates[2] = {"HWC+pad", sizes[2], sizes[0], sizes[1]};
                    candidates[3] = {"CHW+pad", sizes[0], sizes[1], sizes[2]};
                    candidate_count = 4;
                }
                else if (num_dims == 3)
                {
                    candidates[0] = {"CHW", sizes[0], sizes[1], sizes[2]};
                    candidates[1] = {"HWC", sizes[2], sizes[0], sizes[1]};
                    candidate_count = 2;
                }

                std::fprintf(stderr, "AwnnEngine: VIP raw sizes=[");
                for (vip_uint32_t d = 0; d < num_dims; ++d)
                {
                    std::fprintf(stderr, "%s%u", d > 0 ? "," : "", sizes[d]);
                }
                std::fprintf(stderr, "], byte_size=%u\n", byte_size);
                std::fprintf(stderr, "AwnnEngine: possible Param (input_channels, input_height, input_width):\n");

                for (std::size_t i = 0; i < candidate_count; ++i)
                {
                    const Candidate &cand = candidates[i];
                    if (cand.c == 0 || cand.h == 0 || cand.w == 0)
                    {
                        continue;
                    }
                    const std::uint64_t product =
                        static_cast<std::uint64_t>(cand.c) * cand.h * cand.w;
                    std::fprintf(
                        stderr,
                        "  %ux%ux%u  (%s)%s\n",
                        cand.c,
                        cand.h,
                        cand.w,
                        cand.label,
                        product == byte_size ? "  <-- matches buffer" : "");
                }
            }

            bool applyParamInputShape(
                const AwnnEngine::Param &param,
                vip_uint32_t byte_size,
                std::uint32_t &channels,
                std::uint32_t &height,
                std::uint32_t &width)
            {
                if (param.input_channels == 0 || param.input_height == 0 || param.input_width == 0)
                {
                    std::fprintf(
                        stderr,
                        "AwnnEngine: Param input_channels/height/width must be non-zero\n");
                    return false;
                }

                const std::uint64_t product = static_cast<std::uint64_t>(param.input_channels) *
                                              param.input_height * param.input_width;
                if (product != byte_size)
                {
                    std::fprintf(
                        stderr,
                        "AwnnEngine: Param %ux%ux%u (%llu bytes) != buffer %u bytes\n",
                        param.input_channels,
                        param.input_height,
                        param.input_width,
                        static_cast<unsigned long long>(product),
                        byte_size);
                    return false;
                }

                channels = param.input_channels;
                height = param.input_height;
                width = param.input_width;
                return true;
            }

            std::optional<post_process::TensorDtype> dtypeFromFormat(int format)
            {
                if (format == VIP_BUFFER_FORMAT_INT8 || format == VIP_BUFFER_FORMAT_UINT8)
                {
                    return post_process::TensorDtype::INT8;
                }
                if (format == VIP_BUFFER_FORMAT_FP32)
                {
                    return post_process::TensorDtype::FP32;
                }
                return std::nullopt;
            }

        } // namespace

        AwnnEngine::AwnnEngine(const Param &param) : param_(param)
        {
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
            if (vip_create_network(
                    param_.model_path.c_str(),
                    0,
                    VIP_CREATE_NETWORK_FROM_FILE,
                    &network) != VIP_SUCCESS ||
                network == nullptr)
            {
                return false;
            }
            network_ = network;

            if (!prepareIo())
            {
                return false;
            }

            if (vip_prepare_network(network) != VIP_SUCCESS)
            {
                return false;
            }
            network_prepared_ = true;

            for (vip_uint32_t i = 0; i < input_buffers_.size(); ++i)
            {
                if (vip_set_input(network, i, static_cast<vip_buffer>(input_buffers_[i])) != VIP_SUCCESS)
                {
                    return false;
                }
            }

            for (vip_uint32_t i = 0; i < output_buffers_vip_.size(); ++i)
            {
                if (vip_set_output(network, i, static_cast<vip_buffer>(output_buffers_vip_[i])) != VIP_SUCCESS)
                {
                    return false;
                }
            }

            output_data_.assign(output_buffers_vip_.size(), nullptr);
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

            info_ = {};
            input_buffers_.clear();
            output_buffers_vip_.clear();

            info_.input_channels.reserve(input_count);
            info_.input_heights.reserve(input_count);
            info_.input_widths.reserve(input_count);
            info_.input_byte_sizes.reserve(input_count);
            info_.input_dtypes.reserve(input_count);
            info_.output_byte_sizes.reserve(output_count);
            info_.output_dtypes.reserve(output_count);

            for (vip_uint32_t i = 0; i < input_count; ++i)
            {
                vip_buffer_create_params_t param{};
                if (!queryCreateParam(network, i, false, param))
                {
                    return false;
                }

                vip_buffer buffer = nullptr;
                if (vip_create_buffer(&param, 0, &buffer) != VIP_SUCCESS || buffer == nullptr)
                {
                    return false;
                }

                const std::uint32_t byte_size = vip_get_buffer_size(buffer);

                logPossibleInputShapes(param.num_of_dims, param.sizes, byte_size);

                std::uint32_t channels = 0;
                std::uint32_t height = 0;
                std::uint32_t width = 0;
                if (!applyParamInputShape(param_, byte_size, channels, height, width))
                {
                    vip_destroy_buffer(buffer);
                    return false;
                }

                const auto dtype = dtypeFromFormat(param.data_format);
                if (!dtype.has_value())
                {
                    vip_destroy_buffer(buffer);
                    return false;
                }

                input_buffers_.push_back(buffer);
                info_.input_channels.push_back(channels);
                info_.input_heights.push_back(height);
                info_.input_widths.push_back(width);
                info_.input_byte_sizes.push_back(byte_size);
                info_.input_dtypes.push_back(*dtype);
            }

            for (vip_uint32_t i = 0; i < output_count; ++i)
            {
                vip_buffer_create_params_t param{};
                if (!queryCreateParam(network, i, true, param))
                {
                    return false;
                }

                vip_buffer buffer = nullptr;
                if (vip_create_buffer(&param, 0, &buffer) != VIP_SUCCESS || buffer == nullptr)
                {
                    return false;
                }

                const auto dtype = dtypeFromFormat(param.data_format);
                if (!dtype.has_value())
                {
                    vip_destroy_buffer(buffer);
                    return false;
                }

                output_buffers_vip_.push_back(buffer);
                info_.output_byte_sizes.push_back(vip_get_buffer_size(buffer));
                info_.output_dtypes.push_back(*dtype);
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

            info_ = {};
            input_buffers_.clear();
            output_buffers_vip_.clear();
            output_data_.clear();
            valid_ = false;
        }

        bool AwnnEngine::run(const void *input_buffer, std::size_t input_byte_size)
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

            if (vip_run_network(static_cast<vip_network>(network_)) != VIP_SUCCESS)
            {
                return false;
            }

            if (!map_outputs())
            {
                release_outputs_unlocked();
                return false;
            }

            result_.outputs.clear();
            result_.outputs.reserve(info_.output_byte_sizes.size());
            for (std::size_t i = 0; i < info_.output_byte_sizes.size(); ++i)
            {
                post_process::TensorView view{};
                view.byte_size = info_.output_byte_sizes[i];
                view.data = output_data_[i];
                view.dtype = info_.output_dtypes[i];
                result_.outputs.push_back(view);
            }
            result_.ready = !result_.outputs.empty();
            return true;
        }

        void AwnnEngine::releaseResult()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            release_outputs_unlocked();
        }

        void AwnnEngine::release_outputs_unlocked()
        {
            for (std::uint32_t i = 0; i < output_buffers_vip_.size(); ++i)
            {
                if (output_data_[i] != nullptr)
                {
                    vip_unmap_buffer(static_cast<vip_buffer>(output_buffers_vip_[i]));
                    output_data_[i] = nullptr;
                }
            }

            result_.outputs.clear();
            result_.ready = false;
        }

        bool AwnnEngine::map_outputs()
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

                output_data_[i] = out_data;
            }

            return true;
        }

        bool AwnnEngine::copyInput(const void *user_input, std::size_t input_byte_size)
        {
            auto *buffer = static_cast<vip_buffer>(input_buffers_[0]);
            const std::uint32_t buf_size = info_.input_byte_sizes.at(0);
            if (input_byte_size < buf_size)
            {
                return false;
            }

            void *device_data = vip_map_buffer(buffer);
            if (device_data == nullptr)
            {
                return false;
            }

            std::memcpy(device_data, user_input, buf_size);
            vip_unmap_buffer(buffer);

            return vip_flush_buffer(buffer, VIP_BUFFER_OPER_TYPE_FLUSH) == VIP_SUCCESS;
        }

    } // namespace engine
} // namespace deploy_percept

#endif
