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

        bool BaseEngine::get_binary_file_size(const std::string& filepath, size_t& size)
        {
            // 使用RAII管理文件资源
            std::ifstream file(filepath, std::ios::binary);
            if (!file) {
                return false;
            }

            // 获取文件大小
            file.seekg(0, std::ios::end);
            std::streamsize file_size = file.tellg();
            if (file_size <= 0) {
                return false;
            }

            size = static_cast<size_t>(file_size);
            return true;
        }

        bool BaseEngine::load_binary_file_data(const std::string& filepath, std::vector<unsigned char>& data)
        {
            // 使用RAII管理文件资源
            std::ifstream file(filepath, std::ios::binary);
            if (!file) {
                return false;
            }

            // 获取文件大小
            file.seekg(0, std::ios::end);
            std::streamsize file_size = file.tellg();
            if (file_size <= 0) {
                return false;
            }

            // 移动到文件开头
            file.seekg(0, std::ios::beg);

            // 分配内存并读取文件内容
            data.resize(file_size);
            if (!file.read(reinterpret_cast<char*>(data.data()), file_size)) {
                return false; // 读取失败
            }

            return true;
        }

    } // namespace engine
} // namespace deploy_percept