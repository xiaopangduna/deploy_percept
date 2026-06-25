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

        /** 最近一次 run 的分段耗时（毫秒） */
        struct RunTiming
        {
            double npu_ms{0};           ///< vip_run_network
            double output_fetch_ms{0};  ///< Mapped：invalidate+map；HostCopy：map+memcpy+unmap
        };

        /** 构造时指定输出取数策略（整个 engine 实例固定） */
        enum class OutputFetch
        {
            HostCopy, ///< 非零拷贝：memcpy 到 engine 内 host 缓冲（默认，对齐 ai-sdk）
            Mapped,   ///< 零拷贝：map VIP 输出，post 后须 release_outputs()
        };

        /** 当前 output_buffers*() 指向的存储方式（最近一次 run 成功后） */
        enum class OutputStorage
        {
            None,
            Mapped,
            HostCopy,
        };

        /** VIPLite 推理引擎：暴露 raw 输出（INT8/UINT8 或 FP32） */
        class AwnnEngine : public BaseEngine
        {
        public:
            struct Params
            {
                std::string model_path;
                OutputFetch output_fetch{OutputFetch::HostCopy};
            };

            explicit AwnnEngine(const Params &params);
            ~AwnnEngine();

            const Params &getParams() const { return params_; }
            bool is_valid() const { return valid_; }

            /**
             * 推理：输入 copy 到 VIP；输出策略由 Params::output_fetch 决定。
             * HostCopy：output_buffers*() 指向 engine host 缓冲，有效至下次 run。
             * Mapped：borrowed 映射指针，post 后须 release_outputs()。
             */
            bool run(const void *input_buffer, std::size_t input_byte_size);

            /** 仅 OutputFetch::Mapped 时有实际操作（unmap） */
            void release_outputs();
            bool outputs_ready() const { return outputs_ready_; }
            OutputStorage output_storage() const { return output_storage_; }
            const RunTiming &last_run_timing() const { return last_run_timing_; }

            std::uint32_t input_count() const
            {
                return static_cast<std::uint32_t>(input_byte_sizes_.size());
            }
            std::uint32_t input_buffer_byte_size(std::uint32_t index = 0) const
            {
                return input_byte_sizes_.at(index);
            }
            /** 输入 spatial/channel 尺寸（自模型 query，兼容 NCHW / NHWC layout 报告） */
            std::uint32_t input_num_dims(std::uint32_t index = 0) const
            {
                return static_cast<std::uint32_t>(input_attrs_.at(index).dims.size());
            }
            std::uint32_t input_dim(std::uint32_t dim_index, std::uint32_t index = 0) const
            {
                return input_attrs_.at(index).dims.at(dim_index);
            }
            std::uint32_t input_channels(std::uint32_t index = 0) const
            {
                return input_attrs_.at(index).channels;
            }
            std::uint32_t input_height(std::uint32_t index = 0) const
            {
                return input_attrs_.at(index).height;
            }
            std::uint32_t input_width(std::uint32_t index = 0) const
            {
                return input_attrs_.at(index).width;
            }
            std::uint32_t output_buffer_byte_size(std::uint32_t index) const
            {
                return output_byte_sizes_.at(index);
            }
            bool outputs_are_int8() const { return outputs_are_int8_; }
            bool outputs_are_fp32() const { return outputs_are_fp32_; }

            std::uint32_t output_count() const
            {
                return static_cast<std::uint32_t>(output_buffers_vip_.size());
            }
            float *output_float(std::uint32_t index);
            int8_t *output_int8(std::uint32_t index);
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

            bool run_impl(const void *input_buffer, std::size_t input_byte_size, bool copy_outputs_to_host);
            bool copyInput(const void *user_input, std::size_t input_byte_size);
            bool fetch_outputs_mapped();
            bool fetch_outputs_copy();
            void release_outputs_unlocked();

            struct InputAttr
            {
                std::vector<std::uint32_t> dims;
                std::uint32_t channels{0};
                std::uint32_t height{0};
                std::uint32_t width{0};
            };

            Params params_;
            bool valid_{false};
            bool network_prepared_{false};
            void *network_{nullptr};
            mutable std::mutex mutex_;

            std::vector<void *> input_buffers_;
            std::vector<std::uint32_t> input_byte_sizes_;
            std::vector<InputAttr> input_attrs_;
            std::vector<void *> output_buffers_vip_;
            std::vector<std::uint32_t> output_byte_sizes_;
            std::vector<int> output_formats_;

            std::vector<std::vector<uint8_t>> output_raw_storage_;
            std::vector<bool> output_mapped_;
            std::vector<int8_t *> output_ptrs_;
            std::vector<float *> output_float_ptrs_;
            std::vector<float> output_scales_;
            std::vector<int32_t> output_zps_;
            bool outputs_are_int8_{false};
            bool outputs_are_fp32_{false};

            bool outputs_ready_{false};
            OutputStorage output_storage_{OutputStorage::None};
            RunTiming last_run_timing_{};

            static int runtime_ref_count_;
            bool runtime_acquired_{false};
        };

    } // namespace engine
} // namespace deploy_percept

#endif
