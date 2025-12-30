#pragma once

#include <vector>
#include <cstring>
#include <string>

#include "rknn_api.h"

#include "BaseEngine.hpp"

namespace deploy_percept
{
    namespace engine
    {
        class RknnEngine : public BaseEngine
        {
        public:
            // 参数配置结构体
            struct Params
            {
                std::string model_path;
            };

            // 结果结构体
            struct Result
            {
            };

            // 使用参数结构体的构造函数
            explicit RknnEngine(const Params &params);
            ~RknnEngine();
            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            bool run(const rknn_input inputs[], rknn_output outputs[]);
            rknn_context ctx_;
            rknn_input_output_num model_io_num_;
            std::vector<rknn_tensor_attr> model_input_attrs_;
            std::vector<rknn_tensor_attr> model_output_attrs_;

        private:
            Params params_;
            Result result_{};

            std::vector<unsigned char> model_binary_data_;
            size_t model_binary_size_;
        };

    } // namespace post_process
} // namespace deploy_percept
