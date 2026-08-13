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

            void appendTensorDesc(
                SigmastarEngine::Info &info,
                bool is_output,
                const MI_IPU_TensorDesc_t &desc)
            {
                std::array<std::uint32_t, 6> sizes{};
                const std::uint32_t dims =
                    desc.u32TensorDim > 6u ? 6u : desc.u32TensorDim;
                for (std::uint32_t i = 0; i < dims; ++i)
                {
                    sizes[i] = desc.u32TensorShape[i];
                }

                const TensorDtype dtype = dtypeFromElmFormat(desc.eElmFormat);
                const std::uint32_t byte_size = desc.u32BufSize;
                const std::string name = desc.name;

                if (is_output)
                {
                    info.output_num_dims.push_back(desc.u32TensorDim);
                    info.output_sizes.push_back(sizes);
                    info.output_dtypes.push_back(dtype);
                    info.output_elm_formats.push_back(
                        static_cast<std::uint32_t>(desc.eElmFormat));
                    info.output_scalars.push_back(desc.fScalar);
                    info.output_zero_points.push_back(desc.s64ZeroPoint);
                    info.output_byte_sizes.push_back(byte_size);
                    info.output_names.push_back(name);
                }
                else
                {
                    info.input_num_dims.push_back(desc.u32TensorDim);
                    info.input_sizes.push_back(sizes);
                    info.input_dtypes.push_back(dtype);
                    info.input_elm_formats.push_back(
                        static_cast<std::uint32_t>(desc.eElmFormat));
                    info.input_scalars.push_back(desc.fScalar);
                    info.input_zero_points.push_back(desc.s64ZeroPoint);
                    info.input_byte_sizes.push_back(byte_size);
                    info.input_names.push_back(name);
                }
            }

        } // namespace

        int SigmastarRuntime::ref_count_ = 0;
        bool SigmastarRuntime::sys_inited_ = false;
        bool SigmastarRuntime::device_created_ = false;
        std::uint16_t SigmastarRuntime::created_soc_id_ = 0;
        std::uint32_t SigmastarRuntime::created_max_variable_buf_size_ = 0;

        std::uint32_t SigmastarRuntime::queryVariableBufferSize(const std::string &model_path)
        {
            if (model_path.empty())
            {
                return 0;
            }
            auto path = mutableCStr(model_path);
            MI_IPU_OfflineModelStaticInfo_t offline_info{};
            if (MI_IPU_GetOfflineModeStaticInfo(nullptr, path.data(), &offline_info) != MI_SUCCESS)
            {
                return 0;
            }
            return offline_info.u32VariableBufferSize;
        }

        std::uint32_t SigmastarRuntime::resolveMaxVariableBufSize(
            const Params &params,
            bool log_errors)
        {
            std::uint32_t needed = params.max_variable_buf_size;
            for (const auto &path : params.model_paths)
            {
                const std::uint32_t var = queryVariableBufferSize(path);
                if (var == 0)
                {
                    if (log_errors)
                    {
                        std::fprintf(
                            stderr,
                            "SigmastarRuntime: GetOfflineModeStaticInfo failed (%s)\n",
                            path.c_str());
                    }
                    return 0;
                }
                needed = std::max(needed, var);
            }
            return needed;
        }

        SigmastarRuntime::SigmastarRuntime(const Params &params)
        {
            if (ref_count_++ == 0)
            {
                if (!initDevice(params))
                {
                    destroyDevice();
                    --ref_count_;
                    return;
                }
            }
            else
            {
                const std::uint32_t needed = resolveMaxVariableBufSize(params, true);
                if (!params.model_paths.empty() || params.max_variable_buf_size > 0)
                {
                    if (needed == 0)
                    {
                        --ref_count_;
                        return;
                    }
                    if (needed > created_max_variable_buf_size_)
                    {
                        std::fprintf(
                            stderr,
                            "SigmastarRuntime: model var %u exceeds CreateDevice max %u "
                            "(recreate Runtime with all model_paths)\n",
                            needed,
                            created_max_variable_buf_size_);
                        --ref_count_;
                        return;
                    }
                }
            }
            acquired_ = true;
            ok_ = true;
        }

        SigmastarRuntime::~SigmastarRuntime()
        {
            if (!acquired_)
            {
                return;
            }
            acquired_ = false;
            ok_ = false;
            if (ref_count_ > 0 && --ref_count_ == 0)
            {
                destroyDevice();
            }
        }

        bool SigmastarRuntime::initDevice(const Params &params)
        {
            if (MI_SYS_Init(params.soc_id) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarRuntime: MI_SYS_Init failed\n");
                return false;
            }
            sys_inited_ = true;
            created_soc_id_ = params.soc_id;

            const std::uint32_t needed = resolveMaxVariableBufSize(params, true);
            if (needed == 0)
            {
                std::fprintf(
                    stderr,
                    "SigmastarRuntime: need model_paths or max_variable_buf_size "
                    "for CreateDevice\n");
                return false;
            }

            auto firmware_path = mutableCStr(params.firmware_path);
            MI_IPU_DevAttr_t dev_attr{};
            // 共享 scratch：同一时刻只有一个 Invoke，取各模型 var 的 max 而非求和
            dev_attr.u32MaxVariableBufSize = needed;
            dev_attr.u32YUV420_W_Pitch_Alignment = 16;
            dev_attr.u32YUV420_H_Pitch_Alignment = 2;
            dev_attr.u32XRGB_W_Pitch_Alignment = 16;

            if (MI_IPU_CreateDevice(&dev_attr, nullptr, firmware_path.data(), 0) != MI_SUCCESS)
            {
                std::fprintf(
                    stderr,
                    "SigmastarRuntime: CreateDevice failed (firmware=%s var=%u)\n",
                    params.firmware_path.c_str(),
                    needed);
                return false;
            }
            device_created_ = true;
            created_max_variable_buf_size_ = needed;
            return true;
        }

        void SigmastarRuntime::destroyDevice()
        {
            if (device_created_)
            {
                MI_IPU_DestroyDevice();
                device_created_ = false;
            }
            created_max_variable_buf_size_ = 0;
            if (sys_inited_)
            {
                MI_SYS_Exit(created_soc_id_);
                sys_inited_ = false;
            }
            created_soc_id_ = 0;
        }

        SigmastarEngine::SigmastarEngine(const Params &params) : params_(params)
        {
            valid_ = initChannel() && prepareInputBuffer();
            if (!valid_)
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
            if (!valid_ || input_buffer == nullptr || info_.input_byte_sizes.empty())
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
            result_.tensor_views.reserve(info_.output_byte_sizes.size());
            for (std::size_t i = 0; i < info_.output_byte_sizes.size(); ++i)
            {
                TensorView view{};
                view.byte_size = info_.output_byte_sizes[i];
                view.data = output_tensor_vector_.astArrayTensors[i].ptTensorData[0];
                view.dtype = info_.output_dtypes[i];
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
            auto dump_side = [&](const char *tag,
                                 const std::vector<std::array<std::uint32_t, 6>> &sizes,
                                 const std::vector<std::uint32_t> &num_dims,
                                 const std::vector<TensorDtype> &dtypes,
                                 const std::vector<std::uint32_t> &bytes,
                                 const std::vector<std::string> &names,
                                 const std::vector<std::uint32_t> &elm_fmts,
                                 const std::vector<float> &scalars) {
                if (sizes.empty())
                {
                    out += "  (no ";
                    out += tag;
                    out += "s)\n";
                    return;
                }
                for (std::size_t i = 0; i < sizes.size(); ++i)
                {
                    char line[512];
                    const char *name =
                        (i < names.size() && !names[i].empty()) ? names[i].c_str() : "?";
                    const std::uint32_t dims = (i < num_dims.size()) ? num_dims[i] : 0u;
                    const TensorDtype dtype =
                        (i < dtypes.size()) ? dtypes[i] : TensorDtype::FP32;
                    const std::uint32_t n = (i < bytes.size()) ? bytes[i] : 0u;
                    const std::uint32_t elm =
                        (i < elm_fmts.size()) ? elm_fmts[i] : 0u;
                    const float scale = (i < scalars.size()) ? scalars[i] : 0.f;
                    std::snprintf(
                        line,
                        sizeof(line),
                        "  %s[%zu] %s: dims=%u shape=[%u,%u,%u,%u] dtype=%s "
                        "elm=%u scalar=%g bytes=%u\n",
                        tag,
                        i,
                        name,
                        dims,
                        sizes[i][0],
                        sizes[i][1],
                        sizes[i][2],
                        sizes[i][3],
                        dtypeToString(dtype),
                        elm,
                        static_cast<double>(scale),
                        n);
                    out += line;
                }
            };

            dump_side(
                "input",
                info_.input_sizes,
                info_.input_num_dims,
                info_.input_dtypes,
                info_.input_byte_sizes,
                info_.input_names,
                info_.input_elm_formats,
                info_.input_scalars);
            dump_side(
                "output",
                info_.output_sizes,
                info_.output_num_dims,
                info_.output_dtypes,
                info_.output_byte_sizes,
                info_.output_names,
                info_.output_elm_formats,
                info_.output_scalars);
            return out;
        }

        bool SigmastarEngine::initChannel()
        {
            if (params_.model_path.empty())
            {
                std::fprintf(stderr, "SigmastarEngine: empty model_path\n");
                return false;
            }

            if (!SigmastarRuntime::device_ready())
            {
                std::fprintf(
                    stderr,
                    "SigmastarEngine: SigmastarRuntime not initialized (CreateDevice)\n");
                return false;
            }

            const std::uint32_t var =
                SigmastarRuntime::queryVariableBufferSize(params_.model_path);
            if (var == 0)
            {
                std::fprintf(
                    stderr,
                    "SigmastarEngine: GetOfflineModeStaticInfo failed (%s)\n",
                    params_.model_path.c_str());
                return false;
            }
            if (var > SigmastarRuntime::process_max_variable_buf_size())
            {
                std::fprintf(
                    stderr,
                    "SigmastarEngine: model var %u exceeds Runtime max %u "
                    "(%s); pass this .img in SigmastarRuntime model_paths\n",
                    var,
                    SigmastarRuntime::process_max_variable_buf_size(),
                    params_.model_path.c_str());
                return false;
            }

            auto model_path = mutableCStr(params_.model_path);
            MI_IPUChnAttr_t chn_attr{};
            chn_attr.u32InputBufDepth = 0;
            chn_attr.u32OutputBufDepth = 1;

            if (MI_IPU_CreateCHN(&channel_id_, &chn_attr, nullptr, model_path.data()) !=
                MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: CreateCHN failed\n");
                return false;
            }
            channel_created_ = true;

            std::memset(&model_desc_, 0, sizeof(model_desc_));
            if (MI_IPU_GetInOutTensorDesc(channel_id_, &model_desc_) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: GetInOutTensorDesc failed\n");
                return false;
            }

            if (model_desc_.u32InputTensorCount < 1u)
            {
                std::fprintf(stderr, "SigmastarEngine: model has no inputs\n");
                return false;
            }

            return fillInfoFromDesc(model_desc_);
        }

        bool SigmastarEngine::fillInfoFromDesc(const MI_IPU_SubNet_InputOutputDesc_t &desc)
        {
            info_ = {};
            for (MI_U32 i = 0; i < desc.u32InputTensorCount; ++i)
            {
                appendTensorDesc(info_, false, desc.astMI_InputTensorDescs[i]);
            }
            for (MI_U32 i = 0; i < desc.u32OutputTensorCount; ++i)
            {
                appendTensorDesc(info_, true, desc.astMI_OutputTensorDescs[i]);
            }
            return !info_.input_byte_sizes.empty();
        }

        bool SigmastarEngine::prepareInputBuffer()
        {
            input_buf_size_ = info_.input_byte_sizes[0];
            if (input_buf_size_ == 0)
            {
                std::fprintf(stderr, "SigmastarEngine: input byte size is 0\n");
                return false;
            }

            if (MI_SYS_MMA_Alloc(
                    SigmastarRuntime::process_soc_id(),
                    nullptr,
                    input_buf_size_,
                    &input_phy_) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: MMA_Alloc failed\n");
                return false;
            }

            if (MI_SYS_Mmap(input_phy_, input_buf_size_, &input_virt_, TRUE) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: Mmap failed\n");
                MI_SYS_MMA_Free(SigmastarRuntime::process_soc_id(), input_phy_);
                input_phy_ = 0;
                return false;
            }

            std::memset(input_virt_, 0, input_buf_size_);
            return true;
        }

        bool SigmastarEngine::copyInput(const void *user_input, std::size_t input_byte_size)
        {
            if (input_virt_ == nullptr || input_byte_size != input_buf_size_)
            {
                std::fprintf(
                    stderr,
                    "SigmastarEngine: input size mismatch (got %zu, expect %u)\n",
                    input_byte_size,
                    input_buf_size_);
                return false;
            }

            std::memcpy(input_virt_, user_input, input_byte_size);
            if (MI_SYS_FlushInvCache(input_virt_, input_buf_size_) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: FlushInvCache failed\n");
                return false;
            }
            return true;
        }

        bool SigmastarEngine::invokeAndTakeOutputs()
        {
            MI_IPU_TensorVector_t input_vector{};
            input_vector.u32TensorCount = model_desc_.u32InputTensorCount;
            input_vector.astArrayTensors[0].phyTensorAddr[0] = input_phy_;
            input_vector.astArrayTensors[0].ptTensorData[0] = input_virt_;

            std::memset(&output_tensor_vector_, 0, sizeof(output_tensor_vector_));
            if (MI_IPU_GetOutputTensors(channel_id_, &output_tensor_vector_) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: GetOutputTensors failed\n");
                return false;
            }
            outputs_held_ = true;

            if (MI_IPU_Invoke(channel_id_, &input_vector, &output_tensor_vector_) != MI_SUCCESS)
            {
                std::fprintf(stderr, "SigmastarEngine: Invoke failed\n");
                return false;
            }

            if (output_tensor_vector_.u32TensorCount != info_.output_byte_sizes.size())
            {
                std::fprintf(
                    stderr,
                    "SigmastarEngine: output count mismatch (%u vs %zu)\n",
                    output_tensor_vector_.u32TensorCount,
                    info_.output_byte_sizes.size());
                return false;
            }

            return true;
        }

        bool SigmastarEngine::copyMappedOutputsToHost()
        {
            host_output_storage_.assign(info_.output_byte_sizes.size(), {});
            result_.tensor_views.clear();
            result_.tensor_views.reserve(info_.output_byte_sizes.size());

            for (std::size_t i = 0; i < info_.output_byte_sizes.size(); ++i)
            {
                const std::uint32_t n = info_.output_byte_sizes[i];
                void *src = output_tensor_vector_.astArrayTensors[i].ptTensorData[0];
                if (src == nullptr || n == 0)
                {
                    return false;
                }
                host_output_storage_[i].resize(n);
                std::memcpy(host_output_storage_[i].data(), src, n);

                TensorView view{};
                view.data = host_output_storage_[i].data();
                view.byte_size = n;
                view.dtype = info_.output_dtypes[i];
                result_.tensor_views.push_back(view);
            }
            return true;
        }

        void SigmastarEngine::release_outputs()
        {
            if (outputs_held_)
            {
                MI_IPU_PutOutputTensors(channel_id_, &output_tensor_vector_);
                outputs_held_ = false;
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
            valid_ = false;

            if (input_virt_ != nullptr)
            {
                MI_SYS_Munmap(input_virt_, input_buf_size_);
                input_virt_ = nullptr;
            }
            if (input_phy_ != 0)
            {
                MI_SYS_MMA_Free(SigmastarRuntime::process_soc_id(), input_phy_);
                input_phy_ = 0;
            }
            input_buf_size_ = 0;

            if (channel_created_)
            {
                MI_IPU_DestroyCHN(channel_id_);
                channel_created_ = false;
                channel_id_ = 0;
            }
        }

    } // namespace engine
} // namespace deploy_percept

#endif
