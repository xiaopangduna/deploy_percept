#pragma once

#ifdef AWNN_FOUND

#include "deploy_percept/core/engine/BaseEngine.hpp"
#include "deploy_percept/common/types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        // ==================================================================
        // AWNN 推理栈（本头文件）
        //
        //   VipLiteRuntime  — 进程级 NPU：vip_init / vip_destroy（ref_count，多实例共享）
        //   AwnnEngine      — 模型级：加载 .nb、prepare IO、run（每个模型一个实例）
        //
        // 使用约定：
        //   1. 先构造 VipLiteRuntime，再构造 AwnnEngine（可多个 Engine 共用一个 Runtime 窗口）
        //   2. VipLiteRuntime 须最后析构（作用域包住所有 AwnnEngine）
        //   3. 同一进程内多个 VipLiteRuntime 仍只 vip_init 一次（引用计数）
        // ==================================================================

        /**
         * VIPLite 进程级 NPU 运行时（vip_init / vip_destroy）的 RAII 封装。
         * 与 AwnnEngine 配套使用，见本文件顶部说明。
         */
        class VipLiteRuntime
        {
        public:
            VipLiteRuntime();
            ~VipLiteRuntime();

            VipLiteRuntime(const VipLiteRuntime &) = delete;
            VipLiteRuntime &operator=(const VipLiteRuntime &) = delete;
            VipLiteRuntime(VipLiteRuntime &&) = delete;
            VipLiteRuntime &operator=(VipLiteRuntime &&) = delete;

            bool ok() const { return ok_; }

        private:
            bool ok_{false};
            bool acquired_{false};
            static int ref_count_;
        };

        /**
         * Allwinner VIPLite 单模型推理引擎。
         *
         * 前置条件：进程内已成功 vip_init（由同命名空间 VipLiteRuntime 保证）。
         *
         * 典型流程：
         *   VipLiteRuntime runtime;
         *   AwnnEngine engine(params);
         *   engine.run(input, size);
         *   EngineResultGuard guard(engine);
         *   processor.run(guard.getEngineResult().outputs);
         */
        class AwnnEngine : public BaseEngine
        {
        public:
            struct Params
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
                std::vector<TensorDtype> input_dtypes;
                std::vector<std::uint32_t> input_quant_formats;
                std::vector<std::int32_t> input_fixed_point_pos;
                std::vector<float> input_tf_scale;
                std::vector<std::int32_t> input_tf_zero_point;
                std::vector<std::uint32_t> input_byte_sizes;
                std::vector<std::string> input_names;

                std::vector<std::uint32_t> output_num_dims;
                std::vector<std::array<std::uint32_t, 6>> output_sizes;
                std::vector<TensorDtype> output_dtypes;
                std::vector<std::uint32_t> output_quant_formats;
                std::vector<std::int32_t> output_fixed_point_pos;
                std::vector<float> output_tf_scale;
                std::vector<std::int32_t> output_tf_zero_point;
                std::vector<std::uint32_t> output_byte_sizes;
                std::vector<std::string> output_names;
            };

            /** 继承 BaseEngine::Result；AWNN 暂无额外字段 */
            struct Result : BaseEngine::Result
            {
            };

            explicit AwnnEngine(const Params &params);
            ~AwnnEngine() override;

            const Params &getParams() const { return params_; }
            const Info &getInfo() const { return info_; }

            const Result &getResult() const override { return result_; }

            /** 构造是否成功（模型加载、IO 准备、vip_prepare_network） */
            bool is_valid() const { return valid_; }

            /** 拷贝输入 → 推理 → map 输出并填充 getResult() */
            bool run(const void *input_buffer, std::size_t input_byte_size);

            void releaseResult() override;

        private:
            bool createNetwork();
            bool prepareIo();
            void destroyNetwork();

            bool copyInput(const void *user_input, std::size_t input_byte_size);
            bool map_outputs();
            void release_outputs();

            Params params_;
            Info info_{};
            Result result_{};

            bool valid_{false};
            bool network_prepared_{false};
            void *network_{nullptr};

            std::vector<void *> input_buffers_;
            std::vector<void *> output_buffers_vip_;
            std::vector<void *> output_data_;
        };

    } // namespace engine
} // namespace deploy_percept

#endif
