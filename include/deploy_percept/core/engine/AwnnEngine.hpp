#pragma once

#ifdef AWNN_FOUND

#include "deploy_percept/core/engine/BaseEngine.hpp"

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

            /** vip_init 是否成功（或已由同进程其它实例完成 init） */
            bool ok() const { return ok_; }
            /** 同 ok()，供 if (runtime) 使用 */
            explicit operator bool() const { return ok(); }

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
         * 典型流程（默认 is_copy_host，输出在 host）：
         *   VipLiteRuntime runtime;
         *   AwnnEngine engine(params);
         *   engine.run(input, size);
         *   processor.run(engine.getResult().tensor_views);
         *   engine.releaseResult();
         *
         * is_copy_host == true（默认）时 run() 内将输出拷至 host，unmap VIP buffer。
         * is_copy_host == false：零拷贝 mapped，须 EngineResultGuard。
         */
        class AwnnEngine : public BaseEngine
        {
        public:
            struct Params
            {
                /** .nb 模型文件路径（传给 vip_create_network FROM_FILE） */
                std::string model_path;
                /**
                 * true（默认）：run() 内 memcpy 至 host 并 unmap，tensor_views 指向 engine 自有 buffer。
                 * false：tensor_views 指向 VIP mapped 内存，须 EngineResultGuard 或 post 前勿 release。
                 */
                bool is_copy_host = true;
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

            /** prepareIo 完成后可查；用于构造输入 buffer 尺寸与 layout */
            const Info &getInfo() const { return info_; }

            const Result &getResult() const override { return result_; }

            /** 构造是否成功（模型加载、IO 准备、vip_prepare_network） */
            bool ok() const { return ok_; }
            /** 同 ok()，供 if (engine) 使用 */
            explicit operator bool() const { return ok(); }

            /** 格式化 VIP IO 元信息，供日志/调试；不写入 stdout */
            std::string dump_info() const;

            /**
             * 拷贝 host 输入 → vip_run_network → 填充 getResult().tensor_views。
             * is_copy_host==false：views 指向 mapped；is_copy_host==true：views 指向 host 缓存。
             */
            bool run(const void *input_buffer, std::size_t input_byte_size);

            /** 释放输出（unmap VIP 或清空 host 缓存）并清空 result_.tensor_views */
            void releaseResult() override;

        private:
            bool createNetwork();
            bool prepareIo();
            void destroyNetwork();

            bool copyInput(const void *user_input, std::size_t input_byte_size);
            bool map_outputs();
            void release_outputs();
            bool copyMappedOutputsToHost();

            Params params_;
            Info info_{};
            Result result_{};

            bool ok_{false};
            bool network_prepared_{false};
            void *network_{nullptr};

            std::vector<void *> input_buffers_;
            std::vector<void *> output_buffers_vip_;
            std::vector<void *> output_data_;
            /** is_copy_host==true 时持有 run() 输出字节 */
            std::vector<std::vector<std::uint8_t>> host_output_storage_;
        };

    } // namespace engine
} // namespace deploy_percept

#endif
