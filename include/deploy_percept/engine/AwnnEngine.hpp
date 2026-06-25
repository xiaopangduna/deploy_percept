#pragma once

#ifdef AWNN_FOUND

extern "C" {
#include "vip_lite.h"
}

#include "deploy_percept/engine/BaseEngine.hpp"
#include "deploy_percept/engine/TensorDesc.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        class AwnnEngine : public BaseEngine
        {
        public:
            struct Params
            {
                std::string model_path;
            };

            explicit AwnnEngine(const Params &params);
            ~AwnnEngine();

            const Params &getParams() const { return params_; }
            bool is_valid() const { return network_ != nullptr; }

            bool run(void *input_buffer);
            float **output_buffers();

            vip_uint32_t output_count() const
            {
                return static_cast<vip_uint32_t>(output_descs_.size());
            }
            const TensorDesc &output_desc(vip_uint32_t index) const { return output_descs_.at(index); }
            const std::vector<uint8_t> &output_raw(vip_uint32_t index) const
            {
                return output_raw_storage_.at(index);
            }

        private:
            struct IoTensor
            {
                vip_buffer buffer{nullptr};
                TensorDesc desc{};
                vip_buffer_create_params_t create_param{};
            };

            void acquireRuntime();
            void releaseRuntime();

            bool createNetwork();
            bool prepareIo();
            void destroyNetwork();

            bool queryIoDesc(
                vip_uint32_t index,
                bool is_output,
                TensorDesc &desc,
                vip_buffer_create_params_t &create_param) const;

            bool copyInput(void *user_input);
            bool fetchOutputs();
            bool dequantOutputs();

            Params params_;
            vip_network network_{nullptr};
            mutable std::mutex mutex_;

            std::vector<IoTensor> inputs_;
            std::vector<IoTensor> outputs_;

            std::vector<TensorDesc> output_descs_;
            std::vector<std::vector<uint8_t>> output_raw_storage_;
            std::vector<std::vector<float>> output_float_storage_;
            std::vector<float *> output_float_ptrs_;

            static int runtime_ref_count_;
            bool runtime_acquired_{false};
        };

    } // namespace engine
} // namespace deploy_percept

#endif
