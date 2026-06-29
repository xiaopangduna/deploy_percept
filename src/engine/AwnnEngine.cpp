#ifdef AWNN_FOUND

#include "deploy_percept/engine/AwnnEngine.hpp"

extern "C" {
#include "vip_lite.h"
}

#include <algorithm>
#include <cstring>
#include <optional>

namespace deploy_percept
{
    namespace engine
    {

        namespace
        {

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

            bool queryBufferMeta(
                vip_network network,
                vip_uint32_t index,
                bool is_output,
                vip_buffer_create_params_t &param,
                std::string &name)
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

                vip_char_t name_buf[64] = {};
                if (query(network, index, VIP_BUFFER_PROP_NAME, name_buf) != VIP_SUCCESS)
                {
                    return false;
                }
                name = name_buf;
                return true;
            }

            void appendBufferInfo(
                AwnnEngine::Info &info,
                bool is_output,
                const vip_buffer_create_params_t &param,
                std::uint32_t byte_size,
                const std::string &name,
                post_process::TensorDtype dtype)
            {
                std::array<std::uint32_t, 6> sizes{};
                std::copy_n(param.sizes, sizes.size(), sizes.begin());

                if (is_output)
                {
                    info.output_num_dims.push_back(param.num_of_dims);
                    info.output_sizes.push_back(sizes);
                    info.output_dtypes.push_back(dtype);
                    info.output_quant_formats.push_back(static_cast<std::uint32_t>(param.quant_format));
                    info.output_fixed_point_pos.push_back(param.quant_data.dfp.fixed_point_pos);
                    info.output_tf_scale.push_back(param.quant_data.affine.scale);
                    info.output_tf_zero_point.push_back(
                        static_cast<std::int32_t>(param.quant_data.affine.zeroPoint));
                    info.output_byte_sizes.push_back(byte_size);
                    info.output_names.push_back(name);
                }
                else
                {
                    info.input_num_dims.push_back(param.num_of_dims);
                    info.input_sizes.push_back(sizes);
                    info.input_dtypes.push_back(dtype);
                    info.input_quant_formats.push_back(static_cast<std::uint32_t>(param.quant_format));
                    info.input_fixed_point_pos.push_back(param.quant_data.dfp.fixed_point_pos);
                    info.input_tf_scale.push_back(param.quant_data.affine.scale);
                    info.input_tf_zero_point.push_back(
                        static_cast<std::int32_t>(param.quant_data.affine.zeroPoint));
                    info.input_byte_sizes.push_back(byte_size);
                    info.input_names.push_back(name);
                }
            }

            bool prepareOneBuffer(
                vip_network network,
                vip_uint32_t index,
                bool is_output,
                AwnnEngine::Info &info,
                std::vector<void *> &buffers)
            {
                vip_buffer_create_params_t param{};
                std::string name;
                if (!queryBufferMeta(network, index, is_output, param, name))
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

                buffers.push_back(buffer);
                appendBufferInfo(info, is_output, param, vip_get_buffer_size(buffer), name, *dtype);
                return true;
            }

            void destroyVipBuffers(std::vector<void *> &buffers)
            {
                for (void *buf : buffers)
                {
                    if (buf != nullptr)
                    {
                        vip_destroy_buffer(static_cast<vip_buffer>(buf));
                    }
                }
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

            for (vip_uint32_t i = 0; i < input_count; ++i)
            {
                if (!prepareOneBuffer(network, i, false, info_, input_buffers_))
                {
                    return false;
                }
            }

            for (vip_uint32_t i = 0; i < output_count; ++i)
            {
                if (!prepareOneBuffer(network, i, true, info_, output_buffers_vip_))
                {
                    return false;
                }
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

                destroyVipBuffers(input_buffers_);
                destroyVipBuffers(output_buffers_vip_);

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
