#include "deploy_percept/engine/BaseEngine.hpp"
#include <fstream>
#include <vector>

namespace deploy_percept
{
    namespace engine
    {

        bool BaseEngine::get_binary_file_size(const std::string& filepath, size_t& size)
        {
            std::ifstream file(filepath, std::ios::binary | std::ios::ate);
            if (!file) {
                return false;
            }

            std::streamsize file_size = file.tellg();
            if (file_size < 0) {
                return false;
            }

            size = static_cast<size_t>(file_size);
            return true;
        }

        bool BaseEngine::load_binary_file_data(const std::string& filepath, std::vector<unsigned char>& data)
        {
            std::ifstream file(filepath, std::ios::binary | std::ios::ate);
            if (!file) {
                return false;
            }

            std::streamsize file_size = file.tellg();
            if (file_size < 0) {
                return false;
            }

            // 重置文件指针到开头
            file.seekg(0, std::ios::beg);

            // 调整vector大小以容纳整个文件
            data.resize(file_size);

            // 读取文件内容
            if (!file.read(reinterpret_cast<char*>(data.data()), file_size)) {
                return false;
            }

            return true;
        }

    } // namespace engine
} // namespace deploy_percept