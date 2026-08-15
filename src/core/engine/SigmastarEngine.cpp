#ifdef SIGMASTAR_FOUND

#include "deploy_percept/core/engine/SigmastarEngine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        namespace
        {

            TensorDtype dtypeFromElmFormat(MI_IPU_ELEMENT_FORMAT fmt)
            {
                switch (fmt)
                {
                case MI_IPU_FORMAT_FP32:
                    return TensorDtype::FP32;
                case MI_IPU_FORMAT_INT16:
                    return TensorDtype::INT16;
                case MI_IPU_FORMAT_INT8:
                case MI_IPU_FORMAT_U8:
                case MI_IPU_FORMAT_GRAY:
                case MI_IPU_FORMAT_NV12:
                case MI_IPU_FORMAT_ARGB8888:
                case MI_IPU_FORMAT_ABGR8888:
                    return TensorDtype::INT8;
                default:
                    return TensorDtype::INT8;
                }
            }

            const char *dtypeToString(TensorDtype dtype)
            {
                switch (dtype)
                {
                case TensorDtype::FP32:
                    return "FP32";
                case TensorDtype::INT8:
                    return "INT8";
                case TensorDtype::INT16:
                    return "INT16";
                default:
                    return "UNKNOWN";
                }
            }

            std::vector<char> mutableCStr(const std::string &s)
            {
                std::vector<char> buf(s.begin(), s.end());
                buf.push_back('\0');
                return buf;
            }

            /** 读 .img 头中的 VariableBufferSize，不占 IPU Channel。失败返回 0。 */
            std::uint32_t queryVariableBufferSize(const std::string &model_path)
            {
                if (model_path.empty())
                {
                    return 0;
                }
                auto path = mutableCStr(model_path);
                MI_IPU_OfflineModelStaticInfo_t offline_info{};
                if (MI_IPU_GetOfflineModeStaticInfo(nullptr, path.data(), &offline_info) !=
                    MI_SUCCESS)
                {
                    return 0;
                }
                return offline_info.u32VariableBufferSize;
            }

        } // namespace

        bool SigmastarRuntime::resolveMaxVariableBufSize()
        {
            for (const auto &path : params_.model_paths)
            {
                const std::uint32_t var = queryVariableBufferSize(path);
                if (var == 0)
                {
                    std::fprintf(
                        stderr,
                        "SigmastarRuntime: GetOfflineModeStaticInfo failed (%s)\n",
                        path.c_str());
                    return false;
                }
                params_.max_variable_buf_size = std::max(params_.max_variable_buf_size, var);
            }
            if (params_.max_variable_buf_size == 0)
            {
                std::fprintf(
                    stderr,
                    "SigmastarRuntime: need model_paths or max_variable_buf_size "
                    "for CreateDevice\n");
                return false;
            }
            return true;
        }

        SigmastarRuntime::SigmastarRuntime(const Params &params) : params_(params)
        {
            if (!initDevice())
            {
                destroyDevice();
                return;
            }
            ok_ = true;
        }

        SigmastarRuntime::~SigmastarRuntime()
        {
            if (!ok_)
            {
                return;
            }
            ok_ = false;
            destroyDevice();
        }

        bool SigmastarRuntime::initDevice()
        {
            if (MI_SYS_Init(params_.soc_id) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarRuntime: MI_SYS_Init failed\n");
                return false;
            }
            has_sys_ = true;

            if (!resolveMaxVariableBufSize())
            {
                return false;
            }

            auto firmware_path = mutableCStr(params_.firmware_path);
            MI_IPU_DevAttr_t dev_attr{};
            // 共享 scratch：同一时刻只有一个 Invoke，取各模型 var 的 max 而非求和
            dev_attr.u32MaxVariableBufSize = params_.max_variable_buf_size;

            if (MI_IPU_CreateDevice(&dev_attr, nullptr, firmware_path.data(), 0) != MI_SUCCESS)
            {
                std::fprintf(
                    stderr,
                    "SigmastarRuntime: CreateDevice failed (firmware=%s var=%u)\n",
                    params_.firmware_path.c_str(),
                    params_.max_variable_buf_size);
                return false;
            }
            has_device_ = true;
            return true;
        }

        void SigmastarRuntime::destroyDevice()
        {
            if (has_device_)
            {
                MI_IPU_DestroyDevice();
                has_device_ = false;
            }
            if (has_sys_)
            {
                MI_SYS_Exit(params_.soc_id);
                has_sys_ = false;
            }
        }

        SigmastarEngine::SigmastarEngine(SigmastarRuntime &runtime, const Params &params)
            : runtime_(runtime), params_(params)
        {
            ok_ = initChannel() && prepareInputBuffer();
            if (!ok_)
            {
                destroyAll();
            }
        }

        SigmastarEngine::~SigmastarEngine()
        {
            destroyAll();
        }

        bool SigmastarEngine::run(const void *input_buffer, std::size_t input_byte_size)
        {
            if (!ok_ || input_buffer == nullptr || info_.desc.u32InputTensorCount < 1u)
            {
                return false;
            }

            release_outputs();

            if (!copyInput(input_buffer, input_byte_size))
            {
                return false;
            }

            if (!invokeAndTakeOutputs())
            {
                release_outputs();
                return false;
            }

            if (params_.is_copy_host)
            {
                if (!copyMappedOutputsToHost())
                {
                    release_outputs();
                    return false;
                }
                // host 已持有数据，可立即归还 IPU 输出池
                release_outputs();
                return true;
            }

            result_.tensor_views.clear();
            result_.tensor_views.reserve(info_.desc.u32OutputTensorCount);
            for (MI_U32 i = 0; i < info_.desc.u32OutputTensorCount; ++i)
            {
                const auto &t = info_.desc.astMI_OutputTensorDescs[i];
                TensorView view{};
                view.byte_size = t.u32BufSize;
                view.data = output_tensor_vector_.astArrayTensors[i].ptTensorData[0];
                view.dtype = dtypeFromElmFormat(t.eElmFormat);
                result_.tensor_views.push_back(view);
            }
            return true;
        }

        void SigmastarEngine::releaseResult()
        {
            release_outputs();
            result_.tensor_views.clear();
            host_output_storage_.clear();
        }

        std::string SigmastarEngine::dump_info() const
        {
            std::string out = "SigmastarEngine Info:\n";
            {
                char line[64];
                std::snprintf(line, sizeof(line), "  channel_id=%u\n", info_.channel_id);
                out += line;
            }
            auto dump_side = [&](const char *tag,
                                 const MI_IPU_TensorDesc_t *descs,
                                 MI_U32 count) {
                if (count == 0)
                {
                    out += "  (no ";
                    out += tag;
                    out += "s)\n";
                    return;
                }
                for (MI_U32 i = 0; i < count; ++i)
                {
                    char line[512];
                    const auto &t = descs[i];
                    const char *name = t.name[0] != '\0' ? t.name : "?";
                    std::snprintf(
                        line,
                        sizeof(line),
                        "  %s[%u] %s: dims=%u shape=[%u,%u,%u,%u] dtype=%s "
                        "elm=%u scalar=%g bytes=%u\n",
                        tag,
                        i,
                        name,
                        t.u32TensorDim,
                        t.u32TensorShape[0],
                        t.u32TensorShape[1],
                        t.u32TensorShape[2],
                        t.u32TensorShape[3],
                        dtypeToString(dtypeFromElmFormat(t.eElmFormat)),
                        static_cast<unsigned>(t.eElmFormat),
                        static_cast<double>(t.fScalar),
                        t.u32BufSize);
                    out += line;
                }
            };

            dump_side(
                "input",
                info_.desc.astMI_InputTensorDescs,
                info_.desc.u32InputTensorCount);
            dump_side(
                "output",
                info_.desc.astMI_OutputTensorDescs,
                info_.desc.u32OutputTensorCount);
            return out;
        }

        bool SigmastarEngine::initChannel()
        {
            if (params_.model_path.empty())
            {
                std::fprintf(stderr, "SigmastarEngine: empty model_path\n");
                return false;
            }

            if (!runtime_.ok())
            {
                std::fprintf(
                    stderr,
                    "SigmastarEngine: SigmastarRuntime not initialized (CreateDevice)\n");
                return false;
            }

            const std::uint32_t var = queryVariableBufferSize(params_.model_path);
            if (var == 0)
            {
                std::fprintf(
                    stderr,
                    "SigmastarEngine: GetOfflineModeStaticInfo failed (%s)\n",
                    params_.model_path.c_str());
                return false;
            }
            if (var > runtime_.params().max_variable_buf_size)
            {
                std::fprintf(
                    stderr,
                    "SigmastarEngine: model var %u exceeds Runtime max %u "
                    "(%s); pass this .img in SigmastarRuntime model_paths\n",
                    var,
                    runtime_.params().max_variable_buf_size,
                    params_.model_path.c_str());
                return false;
            }

            auto model_path = mutableCStr(params_.model_path);
            MI_IPUChnAttr_t chn_attr{};
            chn_attr.u32InputBufDepth = 0;
            chn_attr.u32OutputBufDepth = 1;

            if (MI_IPU_CreateCHN(&info_.channel_id, &chn_attr, nullptr, model_path.data()) !=
                MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: CreateCHN failed\n");
                return false;
            }
            has_channel_ = true;

            info_.desc = {};
            if (MI_IPU_GetInOutTensorDesc(info_.channel_id, &info_.desc) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: GetInOutTensorDesc failed\n");
                return false;
            }

            if (info_.desc.u32InputTensorCount < 1u)
            {
                std::fprintf(stderr, "SigmastarEngine: model has no inputs\n");
                return false;
            }

            return true;
        }

        bool SigmastarEngine::prepareInputBuffer()
        {
            const MI_U32 input_buf_size = info_.desc.astMI_InputTensorDescs[0].u32BufSize;
            if (input_buf_size == 0)
            {
                std::fprintf(stderr, "SigmastarEngine: input byte size is 0\n");
                return false;
            }

            if (MI_SYS_MMA_Alloc(
                    runtime_.params().soc_id,
                    nullptr,
                    input_buf_size,
                    &input_phy_) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: MMA_Alloc failed\n");
                return false;
            }

            if (MI_SYS_Mmap(input_phy_, input_buf_size, &input_virt_, TRUE) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: Mmap failed\n");
                MI_SYS_MMA_Free(runtime_.params().soc_id, input_phy_);
                input_phy_ = 0;
                return false;
            }

            std::memset(input_virt_, 0, input_buf_size);
            return true;
        }

        bool SigmastarEngine::copyInput(const void *user_input, std::size_t input_byte_size)
        {
            const MI_U32 input_buf_size = info_.desc.astMI_InputTensorDescs[0].u32BufSize;
            if (input_virt_ == nullptr || input_byte_size != input_buf_size)
            {
                std::fprintf(
                    stderr,
                    "SigmastarEngine: input size mismatch (got %zu, expect %u)\n",
                    input_byte_size,
                    input_buf_size);
                return false;
            }

            std::memcpy(input_virt_, user_input, input_byte_size);
            if (MI_SYS_FlushInvCache(input_virt_, input_buf_size) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: FlushInvCache failed\n");
                return false;
            }
            return true;
        }

        bool SigmastarEngine::invokeAndTakeOutputs()
        {
            MI_IPU_TensorVector_t input_vector{};
            input_vector.u32TensorCount = info_.desc.u32InputTensorCount;
            input_vector.astArrayTensors[0].phyTensorAddr[0] = input_phy_;
            input_vector.astArrayTensors[0].ptTensorData[0] = input_virt_;

            std::memset(&output_tensor_vector_, 0, sizeof(output_tensor_vector_));
            if (MI_IPU_GetOutputTensors(info_.channel_id, &output_tensor_vector_) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: GetOutputTensors failed\n");
                return false;
            }
            has_outputs_ = true;

            if (MI_IPU_Invoke(info_.channel_id, &input_vector, &output_tensor_vector_) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: Invoke failed\n");
                return false;
            }

            if (output_tensor_vector_.u32TensorCount != info_.desc.u32OutputTensorCount)
            {
                std::fprintf(
                    stderr,
                    "SigmastarEngine: output count mismatch (%u vs %u)\n",
                    output_tensor_vector_.u32TensorCount,
                    info_.desc.u32OutputTensorCount);
                return false;
            }

            return true;
        }

        bool SigmastarEngine::copyMappedOutputsToHost()
        {
            host_output_storage_.assign(info_.desc.u32OutputTensorCount, {});
            result_.tensor_views.clear();
            result_.tensor_views.reserve(info_.desc.u32OutputTensorCount);

            for (MI_U32 i = 0; i < info_.desc.u32OutputTensorCount; ++i)
            {
                const auto &t = info_.desc.astMI_OutputTensorDescs[i];
                void *src = output_tensor_vector_.astArrayTensors[i].ptTensorData[0];
                if (src == nullptr || t.u32BufSize == 0)
                {
                    return false;
                }
                host_output_storage_[i].resize(t.u32BufSize);
                std::memcpy(host_output_storage_[i].data(), src, t.u32BufSize);

                TensorView view{};
                view.data = host_output_storage_[i].data();
                view.byte_size = t.u32BufSize;
                view.dtype = dtypeFromElmFormat(t.eElmFormat);
                result_.tensor_views.push_back(view);
            }
            return true;
        }

        void SigmastarEngine::release_outputs()
        {
            if (has_outputs_)
            {
                MI_IPU_PutOutputTensors(info_.channel_id, &output_tensor_vector_);
                has_outputs_ = false;
                std::memset(&output_tensor_vector_, 0, sizeof(output_tensor_vector_));
            }

            if (!params_.is_copy_host)
            {
                result_.tensor_views.clear();
            }
        }

        void SigmastarEngine::destroyAll()
        {
            release_outputs();
            result_.tensor_views.clear();
            host_output_storage_.clear();
            ok_ = false;

            if (input_virt_ != nullptr)
            {
                MI_SYS_Munmap(input_virt_, info_.desc.astMI_InputTensorDescs[0].u32BufSize);
                input_virt_ = nullptr;
            }
            if (input_phy_ != 0)
            {
                MI_SYS_MMA_Free(runtime_.params().soc_id, input_phy_);
                input_phy_ = 0;
            }

            if (has_channel_)
            {
                MI_IPU_DestroyCHN(info_.channel_id);
                has_channel_ = false;
                info_.channel_id = 0;
            }
        }

    } // namespace engine
} // namespace deploy_percept

#endif
