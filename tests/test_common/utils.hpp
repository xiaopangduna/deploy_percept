#pragma once

#include "deploy_percept/common/types.hpp"

#include <vector>

/**
 * @brief 比较两个uint8_t向量是否相等
 */
bool isUint8VectorEqual(const std::vector<uint8_t> &expected, const std::vector<uint8_t> &actual);

/**
 * @brief 比较两个 Detection 向量是否在容差范围内相等（class、score、box）
 */
bool isDetectionVectorEqualWithinTolerance(
    const std::vector<deploy_percept::Detection> &expected,
    const std::vector<deploy_percept::Detection> &actual,
    int box_tolerance_px = 2,
    float score_tolerance = 0.05f);

/**
 * @brief GTest 断言：逐条比对 class_id、score、box（带容差）
 */
void expectDetectionsMatchGolden(
    const std::vector<deploy_percept::Detection> &expected,
    const std::vector<deploy_percept::Detection> &actual,
    int box_tolerance_px = 2,
    float score_tolerance = 0.05f);

/**
 * @brief 创建一个 Detection 实例
 */
deploy_percept::Detection MakeDetectResult(
    int class_id, float score, int x1, int y1, int x2, int y2);

bool isUint8VectorEqualWithTolerance(const std::vector<uint8_t> &expected,
                                     const std::vector<uint8_t> &actual,
                                     double tolerance);
