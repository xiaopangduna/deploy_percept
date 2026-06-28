#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        /** 引擎公共工具基类（二进制模型加载等）；不约束 run / 输出 API */
        class BaseEngine
        {
        public:
            virtual ~BaseEngine() = default;

        protected:
            bool get_binary_file_size(const std::string &filepath, std::size_t &size);

            bool load_binary_file_data(const std::string &filepath, std::vector<unsigned char> &data);
        };

    } // namespace engine
} // namespace deploy_percept
