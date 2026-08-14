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
        //   1. 先构造 SigmastarRuntime，再构造 SigmastarEngine（可多个 Engine 共用）
        //   2. SigmastarRuntime 须最后析构（作用域包住所有 SigmastarEngine）
        //   3. 同一进程同时只允许一个 SigmastarRuntime；析构后再构造可重新 CreateDevice
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
                /** IPU firmware 路径（板端默认布局） */
                std::string firmware_path = "/config/dla/ipu_firmware.bin";
                std::uint16_t soc_id = 0;
                /**
                 * CreateDevice 的 u32MaxVariableBufSize。
                 * 0：从 model_paths 查询后取 max；非 0：与查询结果再取 max。
                 */
                std::uint32_t max_variable_buf_size = 0;
                /**
                 * 将要 CreateCHN 的 .img。CreateDevice 前只读文件头，
                 * 用各模型 u32VariableBufferSize 的 max 作为共享 scratch。
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

            /** 进程内 Device 是否已创建（供 Engine 检查） */
            static bool device_ready() { return device_created_; }

            static std::uint16_t process_soc_id() { return created_soc_id_; }
            static std::uint32_t process_max_variable_buf_size()
            {
                return created_max_variable_buf_size_;
            }

        private:
            bool initDevice(const Params &params);
            void destroyDevice();
            static std::uint32_t resolveMaxVariableBufSize(const Params &params);

            bool ok_{false};

            static bool sys_inited_;
            static bool device_created_;
            static std::uint16_t created_soc_id_;
            static std::uint32_t created_max_variable_buf_size_;
        };

        /**
         * Sigmastar MI_IPU 单模型推理引擎（离线 .img）。
         *
         * 前置条件：进程内已成功 CreateDevice（由同命名空间 SigmastarRuntime 保证）。
         *
         * 典型流程（默认 is_copy_host）：
         *   SigmastarRuntime::Params runtime_params;
         *   runtime_params.model_paths = {model_path};
         *   SigmastarRuntime runtime(runtime_params);
         *   SigmastarEngine engine(params);
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
             * IPU IO 元信息（GetInOutTensorDesc）。
             * 同侧各 vector 下标对齐；input_sizes/output_sizes 为 shape 前 6 维
             * （常见布局：shape[1]=H, shape[2]=W，以模型实际 layout 为准）。
             */
            struct Info
            {
                std::vector<std::uint32_t> input_num_dims;
                std::vector<std::array<std::uint32_t, 6>> input_sizes;
                std::vector<TensorDtype> input_dtypes;
                /** MI_IPU_ELEMENT_FORMAT 原值 */
                std::vector<std::uint32_t> input_elm_formats;
                std::vector<float> input_scalars;
                std::vector<std::int64_t> input_zero_points;
                std::vector<std::uint32_t> input_byte_sizes;
                std::vector<std::string> input_names;

                std::vector<std::uint32_t> output_num_dims;
                std::vector<std::array<std::uint32_t, 6>> output_sizes;
                std::vector<TensorDtype> output_dtypes;
                std::vector<std::uint32_t> output_elm_formats;
                std::vector<float> output_scalars;
                std::vector<std::int64_t> output_zero_points;
                std::vector<std::uint32_t> output_byte_sizes;
                std::vector<std::string> output_names;
            };

            struct Result : BaseEngine::Result
            {
            };

            explicit SigmastarEngine(const Params &params);
            ~SigmastarEngine() override;

            SigmastarEngine(const SigmastarEngine &) = delete;
            SigmastarEngine &operator=(const SigmastarEngine &) = delete;

            const Params &getParams() const { return params_; }
            const Info &getInfo() const { return info_; }
            const Result &getResult() const override { return result_; }
            bool is_valid() const { return valid_; }

            std::string dump_info() const;

            /**
             * 拷贝 host 输入 → Invoke → 填充 getResult().tensor_views。
             * 输入须与 getInfo().input_byte_sizes[0] 一致（Phase 1 单输入）。
             */
            bool run(const void *input_buffer, std::size_t input_byte_size);

            void releaseResult() override;

        private:
            bool initChannel();
            bool fillInfoFromDesc(const MI_IPU_SubNet_InputOutputDesc_t &desc);
            bool prepareInputBuffer();
            void destroyAll();

            bool copyInput(const void *user_input, std::size_t input_byte_size);
            bool invokeAndTakeOutputs();
            bool copyMappedOutputsToHost();
            void release_outputs();

            Params params_;
            Info info_{};
            Result result_{};

            bool valid_{false};
            bool channel_created_{false};
            bool outputs_held_{false};

            MI_IPU_CHN channel_id_{0};
            MI_IPU_SubNet_InputOutputDesc_t model_desc_{};

            void *input_virt_{nullptr};
            MI_PHY input_phy_{0};
            MI_U32 input_buf_size_{0};

            MI_IPU_TensorVector_t output_tensor_vector_{};
            std::vector<std::vector<std::uint8_t>> host_output_storage_;
        };

    } // namespace engine
} // namespace deploy_percept

#endif
