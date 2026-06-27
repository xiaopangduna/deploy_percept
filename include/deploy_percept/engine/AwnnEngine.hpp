#pragma once

#ifdef AWNN_FOUND

#include "deploy_percept/engine/BaseEngine.hpp"
#include "deploy_percept/post_process/types.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        /** 构造时指定输出取数策略（整个 engine 实例固定） */
        enum class OutputFetch
        {
            HostCopy, ///< 非零拷贝：memcpy 到 engine 内 host 缓冲（默认，对齐 ai-sdk）
            Mapped,   ///< 零拷贝：map VIP 输出，post 后须 release_output_views()
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
             * HostCopy：借出 views 指向 engine host 缓冲，有效至下次 run。
             * Mapped：借出 mapped VIP 指针，须 release_output_views() 或 OutputAccess 析构。
             */
            bool run(const void *input_buffer, std::size_t input_byte_size);

            /** 归还 borrow_output_views() 借出的视图；Mapped 时 unmap，HostCopy 时 no-op */
            void release_output_views() override;

            /** 输入 spatial/channel 尺寸（自模型 query，兼容 NCHW / NHWC layout 报告） */
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
            std::uint32_t output_count() const
            {
                return static_cast<std::uint32_t>(output_buffers_vip_.size());
            }

            /** run 成功后借出输出 TensorView；有效至 release_output_views() 或下次 run() */
            std::vector<post_process::TensorView> borrow_output_views() const override;

        private:
            /** 最近一次 run 成功后输出视图的存储方式（内部状态） */
            enum class OutputStorage
            {
                None,
                Mapped,
                HostCopy,
            };

            /** 模型全部输出的统一 dtype（prepareIo 时确定） */
            enum class OutputDtype : std::uint8_t
            {
                None,
                Int8,
                Fp32,
            };

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
            static OutputDtype dtypeFromFormat(int format);

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

            std::vector<std::vector<uint8_t>> output_raw_storage_;
            std::vector<bool> output_mapped_;
            std::vector<void *> output_data_;
            OutputDtype output_dtype_{OutputDtype::None};

            bool outputs_ready_{false};
            OutputStorage output_storage_{OutputStorage::None};

            static int runtime_ref_count_;
            bool runtime_acquired_{false};
        };

    } // namespace engine
} // namespace deploy_percept

#endif
