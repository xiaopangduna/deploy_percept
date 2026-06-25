#pragma once

#ifdef AWNN_FOUND

#include "deploy_percept/engine/BaseEngine.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        /** VIPLite 推理引擎：暴露 raw 输出（INT8/UINT8 或 FP32） */
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
            bool is_valid() const { return valid_; }

            bool run(void *input_buffer, std::size_t input_byte_size);

            std::uint32_t input_count() const
            {
                return static_cast<std::uint32_t>(input_byte_sizes_.size());
            }
            std::uint32_t input_buffer_byte_size(std::uint32_t index = 0) const
            {
                return input_byte_sizes_.at(index);
            }
            std::uint32_t output_buffer_byte_size(std::uint32_t index) const
            {
                return output_byte_sizes_.at(index);
            }
            bool outputs_are_int8() const { return outputs_are_int8_; }
            bool outputs_are_fp32() const { return outputs_are_fp32_; }

            std::uint32_t output_count() const
            {
                return static_cast<std::uint32_t>(output_raw_storage_.size());
            }
            int8_t **output_buffers();
            float **output_buffers_float();
            float output_scale(std::uint32_t index) const { return output_scales_.at(index); }
            int32_t output_zero_point(std::uint32_t index) const { return output_zps_.at(index); }

        private:
            void acquireRuntime();
            void releaseRuntime();

            bool createNetwork();
            bool prepareIo();
            void destroyNetwork();

            bool copyInput(void *user_input, std::size_t input_byte_size);
            bool fetchOutputs();

            Params params_;
            bool valid_{false};
            bool network_prepared_{false};
            void *network_{nullptr};
            mutable std::mutex mutex_;

            std::vector<void *> input_buffers_;
            std::vector<std::uint32_t> input_byte_sizes_;
            std::vector<void *> output_buffers_vip_;
            std::vector<std::uint32_t> output_byte_sizes_;
            std::vector<int> output_formats_;

            std::vector<std::vector<uint8_t>> output_raw_storage_;
            std::vector<int8_t *> output_ptrs_;
            std::vector<float *> output_float_ptrs_;
            std::vector<float> output_scales_;
            std::vector<int32_t> output_zps_;
            bool outputs_are_int8_{false};
            bool outputs_are_fp32_{false};

            static int runtime_ref_count_;
            bool runtime_acquired_{false};
        };

    } // namespace engine
} // namespace deploy_percept

#endif
