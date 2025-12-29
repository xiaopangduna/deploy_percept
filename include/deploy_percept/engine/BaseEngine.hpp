#pragma once

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        class BaseEngine
        {
        public:
            // 获取二进制文件大小，返回值表示是否成功
            bool get_binary_file_size(const std::string& filepath, size_t& size);
            
            // 加载二进制文件数据，返回值表示是否成功
            bool load_binary_file_data(const std::string& filepath, std::vector<unsigned char>& data);
        };

    } // namespace engine
} // namespace deploy_percept