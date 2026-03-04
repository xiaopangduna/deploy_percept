#ifndef DEPLOY_PERCEPT_UTILS_NPY_HPP
#define DEPLOY_PERCEPT_UTILS_NPY_HPP

#include <string>
#include <vector>
#include <cnpy.h>

namespace deploy_percept {
namespace utils {

/**
 * @brief 比较两个cnpy::npz_t对象是否完全一致
 * @param npz1 第一个NPZ对象
 * @param npz2 第二个NPZ对象
 * @return bool true表示完全一致，false表示存在差异
 */
bool areNpzObjectsIdentical(const cnpy::npz_t& npz1, const cnpy::npz_t& npz2);

/**
 * @brief 比较两个NPZ文件是否完全一致
 * @param file1 第一个NPZ文件路径
 * @param file2 第二个NPZ文件路径
 * @return bool true表示完全一致，false表示存在差异
 */
bool compareNpzFiles(const std::string& file1, const std::string& file2);

/**
 * @brief 从cnpy::npz_t对象中加载int8_t类型的输出缓冲区
 * @param npz 包含输出数据的NPZ对象
 * @param num_outputs 输出的数量
 * @return std::vector<int8_t*> 包含int8_t输出缓冲区指针的向量
 * @throws std::runtime_error 当键不存在或数据类型不匹配时抛出异常
 */
std::vector<int8_t*> LoadInt8OutputBuffers(const cnpy::npz_t& npz, int num_outputs);

/**
 * @brief 从cnpy::npz_t对象中加载输出缓冲区
 * @param npz 包含输出数据的NPZ对象
 * @param num_outputs 输出的数量
 * @return std::vector<void*> 包含输出缓冲区指针的向量
 */
std::vector<void*> LoadOutputBuffers(const cnpy::npz_t& npz, int num_outputs);

/**
 * @brief 从NPZ文件读取三个int8_t输出数据到vector中
 * @param filepath NPZ文件路径
 * @param success 输出参数，指示读取是否成功
 * @return std::vector<std::vector<int8_t>> 包含三个输出数据的向量
 */
std::vector<std::vector<int8_t>> readNpzFile(const std::string &filepath, bool &success);

} // namespace utils
} // namespace deploy_percept

#endif // DEPLOY_PERCEPT_UTILS_NPY_HPP