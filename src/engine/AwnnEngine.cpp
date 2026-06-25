#ifdef AWNN_FOUND

#include "deploy_percept/engine/AwnnEngine.hpp"
#include "deploy_percept/engine/AwnnTensorDequant.hpp"

#include <cstdio>
#include <cstring>

namespace deploy_percept
{
    namespace engine
    {

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

            if (!createNetwork())
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
            vip_status_e status = vip_create_network(
                params_.model_path.c_str(),
                0,
                VIP_CREATE_NETWORK_FROM_FILE,
                &network_);
            if (status != VIP_SUCCESS || network_ == nullptr)
            {
                std::fprintf(
                    stderr,
                    "AwnnEngine: vip_create_network failed (%d): %s\n",
                    status,
                    params_.model_path.c_str());
                network_ = nullptr;
                return false;
            }

            if (!prepareIo())
            {
                destroyNetwork();
                return false;
            }

            status = vip_prepare_network(network_);
            if (status != VIP_SUCCESS)
            {
                std::fprintf(stderr, "AwnnEngine: vip_prepare_network failed: %d\n", status);
                destroyNetwork();
                return false;
            }

            for (vip_uint32_t i = 0; i < inputs_.size(); ++i)
            {
                status = vip_set_input(network_, i, inputs_[i].buffer);
                if (status != VIP_SUCCESS)
                {
                    std::fprintf(stderr, "AwnnEngine: vip_set_input(%u) failed: %d\n", i, status);
                    destroyNetwork();
                    return false;
                }
            }

            for (vip_uint32_t i = 0; i < outputs_.size(); ++i)
            {
                status = vip_set_output(network_, i, outputs_[i].buffer);
                if (status != VIP_SUCCESS)
                {
                    std::fprintf(stderr, "AwnnEngine: vip_set_output(%u) failed: %d\n", i, status);
                    destroyNetwork();
                    return false;
                }
            }

            output_descs_.clear();
            output_raw_storage_.clear();
            output_float_storage_.clear();
            output_float_ptrs_.clear();
            output_descs_.reserve(outputs_.size());
            output_raw_storage_.resize(outputs_.size());
            output_float_storage_.resize(outputs_.size());
            output_float_ptrs_.resize(outputs_.size(), nullptr);
            for (const auto &out : outputs_)
            {
                output_descs_.push_back(out.desc);
            }

            return true;
        }

        bool AwnnEngine::prepareIo()
        {
            vip_uint32_t input_count = 0;
            vip_uint32_t output_count = 0;
            vip_status_e status = vip_query_network(
                network_, VIP_NETWORK_PROP_INPUT_COUNT, &input_count);
            if (status != VIP_SUCCESS)
            {
                std::fprintf(stderr, "AwnnEngine: query input count failed: %d\n", status);
                return false;
            }
            status = vip_query_network(
                network_, VIP_NETWORK_PROP_OUTPUT_COUNT, &output_count);
            if (status != VIP_SUCCESS)
            {
                std::fprintf(stderr, "AwnnEngine: query output count failed: %d\n", status);
                return false;
            }

            inputs_.clear();
            outputs_.clear();
            inputs_.resize(input_count);
            outputs_.resize(output_count);

            for (vip_uint32_t i = 0; i < input_count; ++i)
            {
                if (!queryIoDesc(i, false, inputs_[i].desc, inputs_[i].create_param))
                {
                    return false;
                }
                inputs_[i].create_param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
                status = vip_create_buffer(&inputs_[i].create_param, 0, &inputs_[i].buffer);
                if (status != VIP_SUCCESS || inputs_[i].buffer == nullptr)
                {
                    std::fprintf(stderr, "AwnnEngine: create input buffer %u failed: %d\n", i, status);
                    return false;
                }
                inputs_[i].desc.byte_size = vip_get_buffer_size(inputs_[i].buffer);
            }

            for (vip_uint32_t i = 0; i < output_count; ++i)
            {
                if (!queryIoDesc(i, true, outputs_[i].desc, outputs_[i].create_param))
                {
                    return false;
                }
                outputs_[i].create_param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
                status = vip_create_buffer(&outputs_[i].create_param, 0, &outputs_[i].buffer);
                if (status != VIP_SUCCESS || outputs_[i].buffer == nullptr)
                {
                    std::fprintf(stderr, "AwnnEngine: create output buffer %u failed: %d\n", i, status);
                    return false;
                }
                outputs_[i].desc.byte_size = vip_get_buffer_size(outputs_[i].buffer);
            }

            return true;
        }

        bool AwnnEngine::queryIoDesc(
            vip_uint32_t index,
            bool is_output,
            TensorDesc &desc,
            vip_buffer_create_params_t &create_param) const
        {
            std::memset(&create_param, 0, sizeof(create_param));
            create_param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;

            const auto query = is_output ? vip_query_output : vip_query_input;
            vip_status_e status = query(
                network_, index, VIP_BUFFER_PROP_DATA_FORMAT, &create_param.data_format);
            if (status != VIP_SUCCESS)
            {
                return false;
            }
            status = query(
                network_, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &create_param.num_of_dims);
            if (status != VIP_SUCCESS)
            {
                return false;
            }
            status = query(
                network_, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, create_param.sizes);
            if (status != VIP_SUCCESS)
            {
                return false;
            }
            status = query(
                network_, index, VIP_BUFFER_PROP_QUANT_FORMAT, &create_param.quant_format);
            if (status != VIP_SUCCESS)
            {
                return false;
            }

            vip_char_t name[256] = {};
            query(network_, index, VIP_BUFFER_PROP_NAME, name);
            desc.name = name;

            switch (create_param.quant_format)
            {
            case VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:
                query(
                    network_,
                    index,
                    VIP_BUFFER_PROP_FIXED_POINT_POS,
                    &create_param.quant_data.dfp.fixed_point_pos);
                desc.fixed_point_pos = create_param.quant_data.dfp.fixed_point_pos;
                break;
            case VIP_BUFFER_QUANTIZE_TF_ASYMM:
                query(
                    network_,
                    index,
                    VIP_BUFFER_PROP_TF_SCALE,
                    &create_param.quant_data.affine.scale);
                query(
                    network_,
                    index,
                    VIP_BUFFER_PROP_TF_ZERO_POINT,
                    &create_param.quant_data.affine.zeroPoint);
                desc.scale = create_param.quant_data.affine.scale;
                desc.zero_point = create_param.quant_data.affine.zeroPoint;
                break;
            default:
                break;
            }

            desc.data_format = create_param.data_format;
            desc.quant_format = create_param.quant_format;
            desc.num_dims = create_param.num_of_dims;
            desc.dims = {};
            desc.element_count = 1;
            for (vip_uint32_t d = 0; d < create_param.num_of_dims; ++d)
            {
                desc.dims[d] = create_param.sizes[d];
                desc.element_count *= create_param.sizes[d];
            }

            AwnnTensorDequant::buildQuantLut(desc);
            return true;
        }

        void AwnnEngine::destroyNetwork()
        {
            if (network_ != nullptr)
            {
                vip_finish_network(network_);

                for (auto &io : inputs_)
                {
                    if (io.buffer != nullptr)
                    {
                        vip_destroy_buffer(io.buffer);
                        io.buffer = nullptr;
                    }
                }
                for (auto &io : outputs_)
                {
                    if (io.buffer != nullptr)
                    {
                        vip_destroy_buffer(io.buffer);
                        io.buffer = nullptr;
                    }
                }

                vip_destroy_network(network_);
                network_ = nullptr;
            }

            inputs_.clear();
            outputs_.clear();
            output_descs_.clear();
            output_raw_storage_.clear();
            output_float_storage_.clear();
            output_float_ptrs_.clear();

            releaseRuntime();
        }

        bool AwnnEngine::run(void *input_buffer)
        {
            if (!is_valid() || input_buffer == nullptr || inputs_.empty())
            {
                return false;
            }

            std::lock_guard<std::mutex> lock(mutex_);

            if (!copyInput(input_buffer))
            {
                return false;
            }

            vip_status_e status = vip_run_network(network_);
            if (status != VIP_SUCCESS)
            {
                std::fprintf(stderr, "AwnnEngine: vip_run_network failed: %d\n", status);
                return false;
            }

            if (!fetchOutputs() || !dequantOutputs())
            {
                return false;
            }

            return true;
        }

        bool AwnnEngine::copyInput(void *user_input)
        {
            void *device_data = vip_map_buffer(inputs_[0].buffer);
            if (device_data == nullptr)
            {
                std::fprintf(stderr, "AwnnEngine: vip_map_buffer(input) failed\n");
                return false;
            }

            const vip_uint32_t buff_size = vip_get_buffer_size(inputs_[0].buffer);
            std::memcpy(device_data, user_input, buff_size);
            vip_unmap_buffer(inputs_[0].buffer);

            if (vip_flush_buffer(inputs_[0].buffer, VIP_BUFFER_OPER_TYPE_FLUSH) != VIP_SUCCESS)
            {
                std::fprintf(stderr, "AwnnEngine: flush input buffer failed\n");
                return false;
            }

            return true;
        }

        bool AwnnEngine::fetchOutputs()
        {
            for (vip_uint32_t i = 0; i < outputs_.size(); ++i)
            {
                if (vip_flush_buffer(outputs_[i].buffer, VIP_BUFFER_OPER_TYPE_INVALIDATE) != VIP_SUCCESS)
                {
                    std::fprintf(stderr, "AwnnEngine: invalidate output %u failed\n", i);
                    return false;
                }

                void *out_data = vip_map_buffer(outputs_[i].buffer);
                if (out_data == nullptr)
                {
                    std::fprintf(stderr, "AwnnEngine: vip_map_buffer(output %u) failed\n", i);
                    return false;
                }

                auto &raw = output_raw_storage_[i];
                raw.resize(outputs_[i].desc.byte_size);
                std::memcpy(raw.data(), out_data, outputs_[i].desc.byte_size);
                vip_unmap_buffer(outputs_[i].buffer);
            }

            return true;
        }

        bool AwnnEngine::dequantOutputs()
        {
            for (vip_uint32_t i = 0; i < outputs_.size(); ++i)
            {
                if (!AwnnTensorDequant::toFloat(
                        output_descs_[i],
                        output_raw_storage_[i].data(),
                        output_raw_storage_[i].size(),
                        output_float_storage_[i]))
                {
                    std::fprintf(stderr, "AwnnEngine: dequant output %u failed\n", i);
                    return false;
                }
                output_float_ptrs_[i] = output_float_storage_[i].data();
            }
            return true;
        }

        float **AwnnEngine::output_buffers()
        {
            if (!is_valid() || output_float_ptrs_.empty())
            {
                return nullptr;
            }
            return output_float_ptrs_.data();
        }

    } // namespace engine
} // namespace deploy_percept

#endif
