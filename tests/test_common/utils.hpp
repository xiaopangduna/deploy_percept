#pragma once

#include "deploy_percept/common/types.hpp"

#include <vector>

/**
 * @brief 比较两个uint8_t向量是否相等
 */
bool isUint8VectorEqual(const std::vector<uint8_t> &expected, const std::vector<uint8_t> &actual);

/**
 * @brief 比较两个Detection向量是否在容差范围内相等
 */
bool isDetectionVectorEqualWithinTolerance(
    const std::vector<deploy_percept::Detection> &expected,
    const std::vector<deploy_percept::Detection> &actual);

/**
 * @brief 创建一个 Detection 实例
 */
deploy_percept::Detection MakeDetectResult(
    int class_id, float score, int x1, int y1, int x2, int y2);

bool isUint8VectorEqualWithTolerance(const std::vector<uint8_t> &expected,
                                     const std::vector<uint8_t> &actual,
                                     double tolerance);
