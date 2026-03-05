#include "deploy_percept/utils/io.hpp"
#include <fstream>

namespace deploy_percept
{
    namespace utils
    {

        std::vector<uint8_t> loadUint8VectorFromBinFile(const std::filesystem::path &file_path)
        {
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                throw std::runtime_error("Cannot open file: " + file_path.string());
            }

            auto size = std::filesystem::file_size(file_path);
            std::vector<uint8_t> res(size);

            if (!file.read(reinterpret_cast<char *>(res.data()), static_cast<std::streamsize>(size)))
            {
                throw std::runtime_error("Failed to read file: " + file_path.string());
            }

            return res;
        }
        bool saveUint8VectorToBinFile(const std::vector<uint8_t> &data,
                                      const std::filesystem::path &file_path)
        {
            std::ofstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                return false;
            }

            auto size = data.size();
            if (size > 0)
            {
                file.write(reinterpret_cast<const char *>(data.data()),
                           static_cast<std::streamsize>(size));
                if (!file.good())
                {
                    return false;
                }
            }
            return true;
        }

    } // namespace utils
} // namespace deploy_percept