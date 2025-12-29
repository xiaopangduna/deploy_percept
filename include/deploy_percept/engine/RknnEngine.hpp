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
                std::string path_model;
            };

            // 结果结构体
            struct Result
            {
            };

            // 使用参数结构体的构造函数
            explicit RknnEngine(const Params &params);
            ~RknnEngine() = default;
            const Params &getParams() const { return params_; }
            const Result &getResult() const { return result_; }

            // bool run(
            //     int8_t *input0,
            //     int8_t *input1,
            //     int8_t *input2,
            //     int model_in_h,
            //     int model_in_w,
            //     BoxRect pads,
            //     float scale_w,
            //     float scale_h,
            //     std::vector<int32_t> &qnt_zps,
            //     std::vector<float> &qnt_scales);

        private:
            Params params_;
            Result result_{};

            rknn_context ctx_;
        };

    } // namespace post_process
} // namespace deploy_percept
