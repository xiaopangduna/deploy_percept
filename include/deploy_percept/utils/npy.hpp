#ifndef DEPLOY_PERCEPT_UTILS_NPY_HPP
#define DEPLOY_PERCEPT_UTILS_NPY_HPP

#include <string>
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

} // namespace utils
} // namespace deploy_percept

#endif // DEPLOY_PERCEPT_UTILS_NPY_HPP