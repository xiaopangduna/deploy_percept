#include "deploy_percept/engine/RknnEngine.hpp"
#include <vector>
#include <algorithm>
#include <set>
#include <cstring>

namespace deploy_percept
{
    namespace engine
    {

        RknnEngine::RknnEngine(const RknnEngine::Params &params)
            : params_(params)
        {
            get_binary_file_size(params_.model_path, model_binary_size_);
            load_binary_file_data(params_.model_path, model_binary_data_);
            rknn_init(&ctx_, model_binary_data_.data(), model_binary_size_, 0, NULL);
            rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &model_io_num_, sizeof(model_io_num_));
            model_input_attrs_.resize(model_io_num_.n_input);
            model_output_attrs_.resize(model_io_num_.n_output);
            // 查询输入属性
            for (uint32_t i = 0; i < model_io_num_.n_input; ++i)
            {
                model_input_attrs_[i].index = i;
                rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(model_input_attrs_[i]), sizeof(rknn_tensor_attr));
            }

            // 查询输出属性
            for (uint32_t i = 0; i < model_io_num_.n_output; ++i)
            {
                model_output_attrs_[i].index = i;
                rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(model_output_attrs_[i]), sizeof(rknn_tensor_attr));
            }
            //   printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
            // 使用 RknnEngine 类中已经获取的属性
            // for (uint32_t i = 0; i < io_num.n_input; i++)
            // {
            //     auto &attr = engine.model_input_attrs_[i];
            //     dump_tensor_attr(&attr);
            // }

            // for (uint32_t i = 0; i < io_num.n_output; i++)
            // {
            //     auto &attr = engine.model_output_attrs_[i];
            //     dump_tensor_attr(&attr);
            // }
        }
        RknnEngine::~RknnEngine()
        {
            rknn_destroy(ctx_);
        }
        bool RknnEngine::run(rknn_input inputs[], rknn_output outputs[])
        {
            rknn_inputs_set(ctx_, model_io_num_.n_input, inputs);
            rknn_run(ctx_, NULL);
            rknn_outputs_get(ctx_, model_io_num_.n_output, outputs, NULL);
            return true;
        }

        void RknnEngine::dump_tensor_attr(rknn_tensor_attr *attr)
        {
            std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
            for (int i = 1; i < attr->n_dims; ++i)
            {
                shape_str += ", " + std::to_string(attr->dims[i]);
            }

            printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
                "type=%s, qnt_type=%s, "
                "zp=%d, scale=%f\n",
                attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
                attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
                get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
        }

    } // namespace engine
} // namespace deploy_percept