#pragma once

#ifdef AWNN_FOUND

#include "deploy_percept/engine/BaseEngine.hpp"
#include "deploy_percept/types.hpp"

#include <array>
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
            };

            /**
             * VIP IO 元信息：prepareIo 时由 vip_query_input/output 填充。
             * 同侧各 vector 下标对齐；input_sizes/output_sizes 顺序为 VIP 规定的 [W,H,C,N]。
             */
            struct Info
            {
                std::vector<std::uint32_t> input_num_dims;
                std::vector<std::array<std::uint32_t, 6>> input_sizes;
                std::vector<post_process::TensorDtype> input_dtypes;
                std::vector<std::uint32_t> input_quant_formats;
                std::vector<std::int32_t> input_fixed_point_pos;
                std::vector<float> input_tf_scale;
                std::vector<std::int32_t> input_tf_zero_point;
                std::vector<std::uint32_t> input_byte_sizes;
                std::vector<std::string> input_names;

                std::vector<std::uint32_t> output_num_dims;
                std::vector<std::array<std::uint32_t, 6>> output_sizes;
                std::vector<post_process::TensorDtype> output_dtypes;
                std::vector<std::uint32_t> output_quant_formats;
                std::vector<std::int32_t> output_fixed_point_pos;
                std::vector<float> output_tf_scale;
                std::vector<std::int32_t> output_tf_zero_point;
                std::vector<std::uint32_t> output_byte_sizes;
                std::vector<std::string> output_names;
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
