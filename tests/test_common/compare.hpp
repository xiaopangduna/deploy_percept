#pragma once

#include <vector>
#include <cstdint>
#include "deploy_percept/post_process/types.hpp"

/**
 * @brief 比较两个分割掩码向量是否相等
 * @param expected 期望的分割掩码数据
 * @param actual 实际的分割掩码数据
 * @return bool 比较结果，true表示相等，false表示不相等
 * @details 逐字节比较两个std::vector<uint8_t>类型的分割掩码，
 *          使用Google Test的EXPECT_*断言进行详细的结果验证
 */
bool CompareSegmentationMaskVectors(const std::vector<uint8_t>& expected,
                                   const std::vector<uint8_t>& actual);

/**
 * @brief 比较两个检测结果向量是否相等
 * @param expected 期望的检测结果
 * @param actual 实际的检测结果
 * @return bool 比较结果，true表示相等，false表示不相等
 * @details 比较DetectionObject结构体的各个字段，包括类别ID、名称、置信度和边界框坐标
 */
bool CompareDetectResultVectors(const std::vector<deploy_percept::post_process::DetectionObject>& expected,
                               const std::vector<deploy_percept::post_process::DetectionObject>& actual);