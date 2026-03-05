#include "utils.hpp"
#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
using namespace deploy_percept::post_process;
deploy_percept::post_process::DetectionObject MakeDetectResult(int cls_id, const char *name_str, float conf,
                                                               int x1, int y1, int x2, int y2)
{
    deploy_percept::post_process::DetectionObject res;
    res.cls_id = cls_id;
    res.prop = conf;
    res.box = {x1, y1, x2, y2};
    strncpy(res.name, name_str, sizeof(res.name) - 1);
    res.name[sizeof(res.name) - 1] = '\0';
    return res;
}

bool isUint8VectorEqual(const std::vector<uint8_t> &expected,
                        const std::vector<uint8_t> &actual)
{
    // 1. 检查大小是否一致
    if (expected.size() != actual.size())
    {
        std::cout << "Size mismatch: expected size = " << expected.size()
                  << ", actual size = " << actual.size() << std::endl;
        return false;
    }

    // 2. 快速检查完全相等
    if (memcmp(expected.data(), actual.data(), expected.size()) == 0)
    {
        return true; // 完全一致，不打印任何信息
    }

    // 3. 大小一致但内容不同，统计一致和不一致个数
    size_t equal_count = 0;
    size_t not_equal_count = 0;

    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (expected[i] == actual[i])
            ++equal_count;
        else
            ++not_equal_count;
    }

    // 4. 打印统计信息并返回 false
    std::cout << "Vectors are not equal: " << equal_count << " equal, "
              << not_equal_count << " unequal elements." << std::endl;
    return false;
}

bool isDetectionObjectVectorEqualWithinTolerance(const std::vector<DetectionObject> &expected,
                                                 const std::vector<DetectionObject> &actual)
{
    if (expected.size() != actual.size())
    {
        return false;
    }

    // 逐元素比较
    for (size_t i = 0; i < expected.size(); ++i)
    {
        const auto &exp = expected[i];
        const auto &act = actual[i];

        if (exp.cls_id != act.cls_id)
        {
            return false;
        }

        if (std::fabs(exp.prop - act.prop) > 0.05f)
        {
            return false;
        }

        if (std::abs(exp.box.left - act.box.left) > 2 ||
            std::abs(exp.box.top - act.box.top) > 2 ||
            std::abs(exp.box.right - act.box.right) > 2 ||
            std::abs(exp.box.bottom - act.box.bottom) > 2)
        {
            return false;
        }

        // name 字段被忽略，不进行比较
    }

    return true;
}

bool isUint8VectorEqualWithTolerance(const std::vector<uint8_t> &expected,
                                     const std::vector<uint8_t> &actual,
                                     double tolerance) // 默认容忍度 3%
{
    // 大小不同则直接认为不一致
    if (expected.size() != actual.size())
        return false;

    size_t size = expected.size();
    if (size == 0)
        return true; // 两个空向量相等

    // 快速检查完全相等
    if (memcmp(expected.data(), actual.data(), size) == 0)
        return true;

    // 3. 大小一致但内容不同，统计一致和不一致个数
    size_t equal_count = 0;
    size_t not_equal_count = 0;

    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (expected[i] == actual[i])
            ++equal_count;
        else
            ++not_equal_count;
    }

    // 4. 打印统计信息并返回 false
    std::cout << "Vectors are not equal: " << equal_count << " equal, "
              << not_equal_count << " unequal elements." << std::endl;
              
    // 计算允许的最大不同个数（向下取整）
    size_t max_allowed = static_cast<size_t>(size * tolerance);

    size_t diff_count = 0;
    for (size_t i = 0; i < size; ++i)
    {
        if (expected[i] != actual[i])
        {
            ++diff_count;
            if (diff_count > max_allowed)
                return false;
        }
    }

    return true;
}