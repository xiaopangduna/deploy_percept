#pragma once

#ifdef SIGMASTAR_FOUND

#include "deploy_percept/core/engine/BaseEngine.hpp"

#include "mi_ipu.h"
#include "mi_sys.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        // ==================================================================
        // Sigmastar 推理栈（本头文件）
        //
        //   SigmastarRuntime — 进程级 IPU：SYS_Init / CreateDevice / DestroyDevice / SYS_Exit
        //   SigmastarEngine  — 模型级：CreateCHN、MMA 输入、run（每个模型一个实例）
        //
        // 使用约定：
        //   1. 先构造 SigmastarRuntime，再构造 SigmastarEngine（传入 Runtime 引用）
        //   2. SigmastarEngine 不得长于传入的 SigmastarRuntime（作用域包住所有 Engine）
        //   3. CreateDevice 为进程级；析构后再构造可重新 CreateDevice。
        //      同时存活多个 Runtime 是否可行取决于 SDK / soc_id，本类不再拦截。
        //
        // CreateDevice 的 u32MaxVariableBufSize 取各模型 VariableBufferSize 的 **max，
        // 不是求和**：该 buffer 是 IPU 推理时的共享 scratch，同一时刻只有一个 Invoke
        // 占用它；权重 ROM 与 IO MMA 是 per-channel，在 CreateCHN 时另占。
        // ==================================================================

        /**
         * MI_IPU 进程级运行时（SYS_Init + CreateDevice）的 RAII 封装。
         * 与 SigmastarEngine 配套使用，见本文件顶部说明。
         */
        class SigmastarRuntime
        {
        public:
            struct Params
            {
                /**
                 * IPU 固件 .bin，传给 MI_IPU_CreateDevice。
                 * 默认 /config/dla/ipu_firmware.bin 为板端常见布局。
                 */
                std::string firmware_path = "/config/dla/ipu_firmware.bin";
                /**
                 * MI_SYS_Init 的芯片/SYS 实例号；后续 MMA_Alloc / MMA_Free 也用此 id。
                 * 单 SoC 保持 0。
                 */
                std::uint16_t soc_id = 0;
                /**
                 * CreateDevice 共享 scratch（u32MaxVariableBufSize）的下限。
                 * 0：完全由 model_paths 各 .img 头 VariableBufferSize 取 max；
                 * 非 0：再与查询结果取 max（预留更大空间，或暂时列不出全部 .img）。
                 * 与 model_paths 至少提供其一，否则 CreateDevice 失败。
                 * Runtime::params() 里该字段为抬升后的生效值。
                 */
                std::uint32_t max_variable_buf_size = 0;
                /**
                 * 本进程将要 CreateCHN 的 .img。此处只读文件头、不占 Channel，
                 * 用各模型 u32VariableBufferSize 的 max（不是求和）作为共享 scratch。
                 * 须列全；漏掉更大的模型会导致后续 Engine 初始化失败。
                 */
                std::vector<std::string> model_paths;
            };

            explicit SigmastarRuntime(const Params &params);
            ~SigmastarRuntime();

            /**
             * 禁止拷贝 / 移动（Rule of Five：自定义析构后须显式处理这四者）。
             * 本对象独占进程级 Device，没有可「移交」的所有权。
             */
            /** 禁止拷贝构造（Runtime b = a）。默认拷贝后两个对象都会在析构时 DestroyDevice。 */
            SigmastarRuntime(const SigmastarRuntime &) = delete;
            /** 禁止拷贝赋值（b = a）。默认赋值不释放左边已有 Device，且与右边双重销毁。 */
            SigmastarRuntime &operator=(const SigmastarRuntime &) = delete;
            /** 禁止移动构造（Runtime b = std::move(a)）。默认移动不把源 ok_ 置 false，两边析构会 double-destroy。 */
            SigmastarRuntime(SigmastarRuntime &&) = delete;
            /** 禁止移动赋值（b = std::move(a)）。同上：源对象所有权未交出，左边原 Device 也无法正确释放。 */
            SigmastarRuntime &operator=(SigmastarRuntime &&) = delete;

            /** SYS_Init + CreateDevice 是否成功 */
            bool ok() const { return ok_; }
            /** 同 ok()，供 if (runtime) 使用 */
            explicit operator bool() const { return ok(); }

            /**
             * CreateDevice 生效配置（内部副本）。
             * max_variable_buf_size 已按 model_paths 就地抬升；调用方传入的 Params 不被改写。
             */
            const Params &params() const { return params_; }

        private:
            bool initDevice();
            void destroyDevice();
            bool resolveMaxVariableBufSize();

            Params params_{};
            bool ok_{false};
            bool sys_inited_{false};
            bool device_created_{false};
        };

        /**
         * Sigmastar MI_IPU 单模型推理引擎（离线 .img）。
         *
         * 前置条件：传入的 SigmastarRuntime 已 ok()。Engine 不得长于该 Runtime。
         *
         * 典型流程（默认 is_copy_host）：
         *   SigmastarRuntime::Params runtime_params;
         *   runtime_params.model_paths = {model_path};
         *   SigmastarRuntime runtime(runtime_params);
         *   SigmastarEngine engine(runtime, params);
         *   engine.run(input, size);
         *   // 读 engine.getResult().tensor_views
         *   engine.releaseResult();
         *
         * is_copy_host == false：tensor_views 指向 IPU GetOutputTensors 映射内存，
         * 须在 post 之后 releaseResult()（或 EngineResultGuard），勿在 Invoke 后立即 Put。
         */
        class SigmastarEngine : public BaseEngine
        {
        public:
            struct Params
            {
                /** 离线模型 .img 路径 */
                std::string model_path;
                /**
                 * true（默认）：run() 内将输出拷至 host 并 PutOutputTensors，
                 * tensor_views 指向 engine 自有 buffer。
                 * false：零拷贝 mapped，须 EngineResultGuard / 显式 releaseResult。
                 */
                bool is_copy_host = true;
            };

            /**
             * 构造后固定的 IO 元信息。desc 为 GetInOutTensorDesc 原样。
             * 常见布局：shape[1]=H, shape[2]=W（以模型实际 layout 为准）。
             */
            struct Info
            {
                MI_IPU_SubNet_InputOutputDesc_t desc{};
            };

            struct Result : BaseEngine::Result
            {
            };

            explicit SigmastarEngine(SigmastarRuntime &runtime, const Params &params);
            ~SigmastarEngine() override;

            SigmastarEngine(const SigmastarEngine &) = delete;
            SigmastarEngine &operator=(const SigmastarEngine &) = delete;
            SigmastarEngine(SigmastarEngine &&) = delete;
            SigmastarEngine &operator=(SigmastarEngine &&) = delete;

            const Params &getParams() const { return params_; }
            const Info &getInfo() const { return info_; }
            const Result &getResult() const override { return result_; }
            /** 构造是否成功（CreateCHN、MMA 输入） */
            bool ok() const { return ok_; }
            /** 同 ok()，供 if (engine) 使用 */
            explicit operator bool() const { return ok(); }

            std::string dump_info() const;

            /**
             * 拷贝 host 输入 → Invoke → 填充 getResult().tensor_views。
             * 输入须与 getInfo().desc.astMI_InputTensorDescs[0].u32BufSize 一致（Phase 1 单输入）。
             */
            bool run(const void *input_buffer, std::size_t input_byte_size);

            void releaseResult() override;

        private:
            bool initChannel();
            bool prepareInputBuffer();
            void destroyAll();

            bool copyInput(const void *user_input, std::size_t input_byte_size);
            bool invokeAndTakeOutputs();
            bool copyMappedOutputsToHost();
            void release_outputs();

            SigmastarRuntime &runtime_;
            Params params_;
            Info info_{};
            Result result_{};

            bool ok_{false};
            bool channel_created_{false};
            bool outputs_held_{false};

            MI_IPU_CHN channel_id_{0};

            void *input_virt_{nullptr};
            MI_PHY input_phy_{0};

            MI_IPU_TensorVector_t output_tensor_vector_{};
            std::vector<std::vector<std::uint8_t>> host_output_storage_;
        };

        /** 从 desc 抄输出 shape 前 6 维，供后处理 Params.output_shapes */
        inline std::vector<std::array<std::uint32_t, 6>> ipu_output_shapes(
            const MI_IPU_SubNet_InputOutputDesc_t &desc)
        {
            std::vector<std::array<std::uint32_t, 6>> out(desc.u32OutputTensorCount);
            for (MI_U32 i = 0; i < desc.u32OutputTensorCount; ++i)
            {
                const auto &t = desc.astMI_OutputTensorDescs[i];
                const MI_U32 dims = t.u32TensorDim > 6u ? 6u : t.u32TensorDim;
                for (MI_U32 d = 0; d < dims; ++d)
                {
                    out[i][d] = t.u32TensorShape[d];
                }
            }
            return out;
        }

        /** 从 desc 抄 fScalar，供后处理 Params.output_scalars */
        inline std::vector<float> ipu_output_scalars(const MI_IPU_SubNet_InputOutputDesc_t &desc)
        {
            std::vector<float> out(desc.u32OutputTensorCount);
            for (MI_U32 i = 0; i < desc.u32OutputTensorCount; ++i)
            {
                out[i] = desc.astMI_OutputTensorDescs[i].fScalar;
            }
            return out;
        }

    } // namespace engine
} // namespace deploy_percept

#endif
