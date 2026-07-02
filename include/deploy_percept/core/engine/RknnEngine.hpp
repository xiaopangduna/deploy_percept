#pragma once

#if RKNN_FOUND
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
            struct Params
            {
                std::string model_path;
            };

            explicit RknnEngine(const Params &params);
            ~RknnEngine();
            const Params &getParams() const { return params_; }

            bool run(rknn_input inputs[], rknn_output outputs[]);

            rknn_context ctx_;
            rknn_input_output_num model_io_num_;
            std::vector<rknn_tensor_attr> model_input_attrs_;
            std::vector<rknn_tensor_attr> model_output_attrs_;

            static void dump_tensor_attr(rknn_tensor_attr *attr);

        private:
            Params params_;

            std::vector<unsigned char> model_binary_data_;
            size_t model_binary_size_;
        };

    } // namespace engine
} // namespace deploy_percept

#endif
