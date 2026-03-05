#pragma once

#include <vector>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace deploy_percept
{
    namespace utils
    {

        /**
         * @brief 从二进制文件加载分割掩码数据
         * @param file_path 文件路径
         * @return std::vector<uint8_t> 加载的分割掩码数据
         * @throws std::runtime_error 当文件无法打开或读取失败时抛出异常
         * @details 以二进制模式读取文件，将整个文件内容加载到vector中
         *          支持大文件读取，使用std::ios::ate定位到文件末尾获取大小
         */
        std::vector<uint8_t> loadUint8VectorFromBinFile(const std::filesystem::path &file_path);

        bool saveUint8VectorToBinFile(const std::vector<uint8_t> &data, const std::filesystem::path &file_path);

    } // namespace utils
} // namespace deploy_percept