#pragma once

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace deploy_percept
{
    namespace engine
    {

        class BaseEngine
        {
        public:
            // 加载整个文件或从指定偏移加载指定大小的数据
            static std::unique_ptr<unsigned char[]> load_file_data(const std::string& filepath, 
                                                                 size_t offset = 0, 
                                                                 size_t size = 0);
        };

    } // namespace engine
} // namespace deploy_percept