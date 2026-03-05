#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include "deploy_percept/post_process/types.hpp"

/**
 * @brief 比较两个分割掩码向量是否相等
 * @param expected 期望的分割掩码数据
 * @param actual 实际的分割掩码数据
 * @return bool 比较结果，true表示相等，false表示不相等
 * @details 逐字节比较两个std::vector<uint8_t>类型的分割掩码，
 *          使用Google Test的EXPECT_*断言进行详细的结果验证
 */
bool isUint8VectorEqual(const std::vector<uint8_t> &expected,
                        const std::vector<uint8_t> &actual);

/**
 * @brief 比较两个检测结果向量是否相等
 * @param expected 期望的检测结果
 * @param actual 实际的检测结果
 * @return bool 比较结果，true表示相等，false表示不相等
 * @details 比较DetectionObject结构体的各个字段，包括类别ID、名称、置信度和边界框坐标
 */
bool isDetectionObjectVectorEqualWithinTolerance(const std::vector<deploy_percept::post_process::DetectionObject> &expected,
                                                 const std::vector<deploy_percept::post_process::DetectionObject> &actual);

/**
 * @brief 创建一个DetectionObject实例
 * @param cls_id 类别ID
 * @param name_str 类别名称字符串
 * @param conf 置信度
 * @param x1 边界框左上角x坐标
 * @param y1 边界框左上角y坐标
 * @param x2 边界框右下角x坐标
 * @param y2 边界框右下角y坐标
 * @return DetectionObject 创建的检测对象
 */
deploy_percept::post_process::DetectionObject MakeDetectResult(int cls_id, const char *name_str, float conf,
                                                               int x1, int y1, int x2, int y2);

/**
 * @brief 判断两个 uint8_t 向量是否一致，允许一定比例的元素不同
 * @param expected 期望向量
 * @param actual   实际向量
 * @param tolerance 容忍度比例，1.0 表示允许所有元素不同（100%），0.03 表示允许 3% 的元素不同，默认 0.03
 * @return 若两个向量大小相同且不同元素个数 ≤ size * tolerance，返回 true；否则 false
 */
bool isUint8VectorEqualWithTolerance(const std::vector<uint8_t> &expected,
                                     const std::vector<uint8_t> &actual,
                                     double tolerance = 0.03); 