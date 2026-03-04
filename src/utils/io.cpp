#include "deploy_percept/utils/io.hpp"
#include <fstream>

namespace deploy_percept {
namespace utils {

std::vector<uint8_t> LoadSegmentationResult(const std::filesystem::path &file_path)
{
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + file_path.string());
    }

    std::streamsize size = file.tellg();
    if (size < 0)
    {
        throw std::runtime_error("Failed to get file size: " + file_path.string());
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> seg_mask;
    seg_mask.resize(static_cast<size_t>(size));

    if (!file.read(reinterpret_cast<char *>(seg_mask.data()), size))
    {
        throw std::runtime_error("Failed to read file: " + file_path.string());
    }

    // 可选：检查是否完整读取
    if (file.gcount() != size)
    {
        throw std::runtime_error("Read incomplete: " + file_path.string());
    }

    return seg_mask;
}

} // namespace utils
} // namespace deploy_percept