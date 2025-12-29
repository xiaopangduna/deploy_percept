#include "deploy_percept/engine/BaseEngine.hpp"
#include <memory>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>

namespace deploy_percept
{
    namespace engine
    {

        std::unique_ptr<unsigned char[]> BaseEngine::load_file_data(const std::string& filepath, 
                                                                 size_t offset, 
                                                                 size_t size)
        {
            // 使用RAII管理文件资源
            std::ifstream file(filepath, std::ios::binary | std::ios::ate);
            if (!file) {
                return nullptr;
            }

            // 获取文件大小
            std::streamsize file_size = file.tellg();
            if (file_size <= 0) {
                return nullptr;
            }

            // 如果size为0，表示加载整个文件（从offset到文件末尾）
            std::streamsize read_size = size == 0 ? (file_size - offset) : static_cast<std::streamsize>(size);
            
            // 检查偏移量和读取大小是否有效
            if (offset + read_size > file_size) {
                return nullptr;
            }

            // 移动到指定偏移位置
            file.seekg(offset, std::ios::beg);

            // 分配内存并读取文件内容
            auto data = std::make_unique<unsigned char[]>(read_size);
            if (!data) {
                return nullptr;
            }

            if (!file.read(reinterpret_cast<char*>(data.get()), read_size)) {
                return nullptr; // 读取失败
            }

            return data;
        }

    } // namespace engine
} // namespace deploy_percept