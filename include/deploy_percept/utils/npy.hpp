#pragma once

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
 * @brief 从 npz 文件中读取一组前缀+数字索引的数组，每个数组的数据类型必须为 int8_t。
 * 
 * @param npz   已加载的 cnpy::npz_t 对象（只读引用）
 * @param prefix 数组名称前缀，如 "output"
 * @param count  需要读取的数组个数，将读取 prefix0, prefix1, ..., prefix(count-1)
 * @return std::vector<std::vector<int8_t>>  外层 vector 大小等于 count，每个内层 vector 存储对应数组的数据
 * @throws std::runtime_error 如果某个数组不存在或类型不匹配
 */
std::vector<std::vector<int8_t>> convertNpzToInt8VectorsByPrefix(const cnpy::npz_t &npz,const std::string &prefix,size_t count);

bool save_npz(const std::string& filename, const cnpy::npz_t& npz_data);

} // namespace utils
} // namespace deploy_percept

