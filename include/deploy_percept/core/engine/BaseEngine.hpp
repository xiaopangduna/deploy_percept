#pragma once

#include "deploy_percept/common/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        /** 引擎基类：推理输出数据结构 + 释放协议（不约束 run 签名，不持有运行态成员） */
        class BaseEngine
        {
        public:
            /** 最近一次 run 的 raw 输出；outputs 为非 owning 视图，有效至 releaseResult() 或下次 run() */
            struct Result
            {
                bool ready{false};
                std::vector<TensorView> outputs;
            };

            virtual ~BaseEngine() = default;

            virtual const Result &getResult() const = 0;
            virtual void releaseResult() = 0;

        protected:
            bool get_binary_file_size(const std::string &filepath, std::size_t &size);

            bool load_binary_file_data(const std::string &filepath, std::vector<unsigned char> &data);
        };

    } // namespace engine
} // namespace deploy_percept
