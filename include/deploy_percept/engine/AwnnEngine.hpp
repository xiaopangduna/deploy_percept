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

        /**
         * Allwinner VIPLite 推理引擎。
         *
         * 前置条件：进程内已成功 vip_init（通常由 app 层 VipLiteRuntime RAII 保证；
         * VipLiteRuntime 须在 AwnnEngine 之前构造、之后析构）。
         *
         * 典型流程：
         *   AwnnEngine engine(param);
         *   engine.run(input, size);
         *   post(engine.getResult().outputs, ...);
         *   engine.releaseResult();   // 或 AwnnResultGuard RAII
         *
         * getResult().outputs 为 mapped 内存的非 owning 视图，有效至 releaseResult() 或下次 run()。
         */
        class AwnnEngine : public BaseEngine
        {
        public:
            /** 构造参数 */
            struct Param
            {
                std::string model_path;
                /** 逻辑 NCHW，须满足 input_channels×input_height×input_width == VIP buffer 字节数 */
                std::uint32_t input_channels{0};
                std::uint32_t input_height{0};
                std::uint32_t input_width{0};
            };

            /**
             * 模型 IO 静态元信息（prepareIo 时从 VIP 解码）。
             * 初始化成功后只读；各 vector 按下标对应第 i 路 input/output。
             * 输入 C/H/W 来自 Param（init 时与 VIP byte_size 校验）。
             */
            struct Info
            {
                std::vector<std::uint32_t> input_channels;
                std::vector<std::uint32_t> input_heights;
                std::vector<std::uint32_t> input_widths;
                std::vector<std::uint32_t> input_byte_sizes;
                std::vector<post_process::TensorDtype> input_dtypes;

                std::vector<std::uint32_t> output_byte_sizes;
                std::vector<post_process::TensorDtype> output_dtypes;
            };

            /** 最近一次 run 的 raw 输出；outputs 指向 VIP mapped 内存，不拥有数据 */
            struct Result
            {
                bool ready{false};
                std::vector<post_process::TensorView> outputs;
            };

            explicit AwnnEngine(const Param &param);
            ~AwnnEngine();

            const Param &getParam() const { return param_; }
            const Info &getInfo() const { return info_; }
            const Result &getResult() const { return result_; }

            /** 构造是否成功（模型加载、IO 准备、vip_prepare_network） */
            bool is_valid() const { return valid_; }

            /** 拷贝输入 → 推理 → map 输出并填充 getResult() */
            bool run(const void *input_buffer, std::size_t input_byte_size);

            /** unmap 输出并清空 getResult()；析构与下次 run() 前也会调用 */
            void releaseResult();

        private:
            bool createNetwork();
            bool prepareIo();
            void destroyNetwork();

            bool copyInput(const void *user_input, std::size_t input_byte_size);
            bool map_outputs();
            void release_outputs_unlocked();

            Param param_;
            Info info_{};
            Result result_{};

            bool valid_{false};
            bool network_prepared_{false};
            void *network_{nullptr};
            mutable std::mutex mutex_;

            std::vector<void *> input_buffers_;
            std::vector<void *> output_buffers_vip_;
            std::vector<void *> output_data_;
        };

    } // namespace engine
} // namespace deploy_percept

#endif
