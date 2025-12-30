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
        bool RknnEngine::run(const rknn_input inputs[], rknn_output outputs[])
        {
            // 重置结果

            return true;
        }

    } // namespace engine
} // namespace deploy_percept