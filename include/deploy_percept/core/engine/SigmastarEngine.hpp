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

        /**
         * Sigmastar MI_IPU 单模型推理引擎（离线 .img）。
         *
         * Phase 1：ctor 内完成 SYS_Init / CreateDevice / CreateCHN / MMA 输入缓冲；
         * 无独立 Runtime（CreateDevice 依赖模型 variable buffer size）。
         *
         * 典型流程（默认 is_copy_host）：
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
                /** IPU firmware 路径（板端默认布局） */
                std::string firmware_path = "/config/dla/ipu_firmware.bin";
                std::uint16_t soc_id = 0;
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
            bool initDeviceAndChannel();
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
            bool sys_inited_{false};
            bool device_created_{false};
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
