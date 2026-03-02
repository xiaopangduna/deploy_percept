#include "deploy_percept/utils/npy.hpp"
#include <iostream>
#include <filesystem>
#include <set>
#include <cstring>

namespace deploy_percept {
namespace utils {

bool areNpzObjectsIdentical(const cnpy::npz_t& npz1, const cnpy::npz_t& npz2) {
    try {
        // 1. 比较键的数量
        if (npz1.size() != npz2.size()) {
            std::cout << "NPZ key count mismatch: " << npz1.size() << " vs " << npz2.size() << std::endl;
            return false;
        }
        
        // 2. 比较键名集合
        std::set<std::string> keys1, keys2;
        for (const auto& pair : npz1) keys1.insert(pair.first);
        for (const auto& pair : npz2) keys2.insert(pair.first);
        
        if (keys1 != keys2) {
            std::cout << "NPZ key names mismatch" << std::endl;
            return false;
        }
        
        // 3. 逐个比较每个键对应的数据
        for (const auto& key : keys1) {
            const cnpy::NpyArray& arr1 = npz1.at(key);
            const cnpy::NpyArray& arr2 = npz2.at(key);
            
            // 比较数据类型
            if (arr1.word_size != arr2.word_size) {
                std::cout << "Data type mismatch for key '" << key << "': " 
                         << arr1.word_size << " vs " << arr2.word_size << std::endl;
                return false;
            }
            
            // 比较形状
            if (arr1.shape != arr2.shape) {
                std::cout << "Shape mismatch for key '" << key << "'" << std::endl;
                return false;
            }
            
            // 比较数据大小
            if (arr1.num_vals != arr2.num_vals) {
                std::cout << "Data size mismatch for key '" << key << "': " 
                         << arr1.num_vals << " vs " << arr2.num_vals << std::endl;
                return false;
            }
            
            // 逐字节比较数据内容 - 使用正确的数据访问方式
            if (arr1.word_size == sizeof(int8_t)) {
                const int8_t* data1 = arr1.data<int8_t>();
                const int8_t* data2 = arr2.data<int8_t>();
                if (std::memcmp(data1, data2, arr1.num_vals * sizeof(int8_t)) != 0) {
                    std::cout << "Data content mismatch for key '" << key << "'" << std::endl;
                    return false;
                }
            } else if (arr1.word_size == sizeof(float)) {
                const float* data1 = arr1.data<float>();
                const float* data2 = arr2.data<float>();
                if (std::memcmp(data1, data2, arr1.num_vals * sizeof(float)) != 0) {
                    std::cout << "Data content mismatch for key '" << key << "'" << std::endl;
                    return false;
                }
            } else {
                // 对于其他数据类型，逐元素比较
                const char* data1 = reinterpret_cast<const char*>(arr1.data_holder.get());
                const char* data2 = reinterpret_cast<const char*>(arr2.data_holder.get());
                size_t total_bytes = arr1.num_vals * arr1.word_size;
                if (std::memcmp(data1, data2, total_bytes) != 0) {
                    std::cout << "Data content mismatch for key '" << key << "'" << std::endl;
                    return false;
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error comparing NPZ objects: " << e.what() << std::endl;
        return false;
    }
}

bool compareNpzFiles(const std::string& file1, const std::string& file2) {
    try {
        if (!std::filesystem::exists(file1)) {
            std::cerr << "File not found: " << file1 << std::endl;
            return false;
        }
        if (!std::filesystem::exists(file2)) {
            std::cerr << "File not found: " << file2 << std::endl;
            return false;
        }
        
        cnpy::npz_t npz1 = cnpy::npz_load(file1);
        cnpy::npz_t npz2 = cnpy::npz_load(file2);
        
        return areNpzObjectsIdentical(npz1, npz2);
    } catch (const std::exception& e) {
        std::cerr << "Error comparing NPZ files: " << e.what() << std::endl;
        return false;
    }
}

std::vector<void*> LoadOutputBuffers(const cnpy::npz_t& npz, int num_outputs)
{
    std::vector<void *> buffers;
    buffers.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i)
    {
        std::string key = "output_" + std::to_string(i);
        auto it = npz.find(key);
        if (it == npz.end())
        {
            throw std::runtime_error("Key not found: " + key);
        }
        // 添加 const_cast 以匹配非常量指针
        buffers.push_back(const_cast<void *>(it->second.data<void>()));
    }
    return buffers;
}

} // namespace utils
} // namespace deploy_percept