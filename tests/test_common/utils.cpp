#include "utils.hpp"
#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <iostream>

deploy_percept::Detection MakeDetectResult(
    int class_id, float score, int x1, int y1, int x2, int y2)
{
    deploy_percept::Detection res;
    res.class_id = class_id;
    res.score = score;
    res.box = {x1, y1, x2, y2};
    return res;
}

bool isUint8VectorEqual(const std::vector<uint8_t> &expected,
                        const std::vector<uint8_t> &actual)
{
    if (expected.size() != actual.size())
    {
        std::cout << "Size mismatch: expected size = " << expected.size()
                  << ", actual size = " << actual.size() << std::endl;
        return false;
    }

    if (memcmp(expected.data(), actual.data(), expected.size()) == 0)
    {
        return true;
    }

    size_t equal_count = 0;
    size_t not_equal_count = 0;

    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (expected[i] == actual[i])
            ++equal_count;
        else
            ++not_equal_count;
    }

    std::cout << "Vectors are not equal: " << equal_count << " equal, "
              << not_equal_count << " unequal elements." << std::endl;
    return false;
}

bool isDetectionVectorEqualWithinTolerance(
    const std::vector<deploy_percept::Detection> &expected,
    const std::vector<deploy_percept::Detection> &actual,
    int box_tolerance_px,
    float score_tolerance)
{
    if (expected.size() != actual.size())
    {
        return false;
    }

    for (size_t i = 0; i < expected.size(); ++i)
    {
        const auto &exp = expected[i];
        const auto &act = actual[i];

        if (exp.class_id != act.class_id)
        {
            return false;
        }

        if (std::fabs(exp.score - act.score) > score_tolerance)
        {
            return false;
        }

        if (std::abs(exp.box.left - act.box.left) > box_tolerance_px ||
            std::abs(exp.box.top - act.box.top) > box_tolerance_px ||
            std::abs(exp.box.right - act.box.right) > box_tolerance_px ||
            std::abs(exp.box.bottom - act.box.bottom) > box_tolerance_px)
        {
            return false;
        }
    }

    return true;
}

void expectDetectionsMatchGolden(
    const std::vector<deploy_percept::Detection> &expected,
    const std::vector<deploy_percept::Detection> &actual,
    int box_tolerance_px,
    float score_tolerance)
{
    ASSERT_EQ(expected.size(), actual.size());

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const auto &exp = expected[i];
        const auto &act = actual[i];

        SCOPED_TRACE("detection index " + std::to_string(i));

        EXPECT_EQ(exp.class_id, act.class_id);
        EXPECT_NEAR(exp.score, act.score, score_tolerance) << "score";

        EXPECT_LE(std::abs(exp.box.left - act.box.left), box_tolerance_px) << "box.left";
        EXPECT_LE(std::abs(exp.box.top - act.box.top), box_tolerance_px) << "box.top";
        EXPECT_LE(std::abs(exp.box.right - act.box.right), box_tolerance_px) << "box.right";
        EXPECT_LE(std::abs(exp.box.bottom - act.box.bottom), box_tolerance_px) << "box.bottom";
    }
}

bool isUint8VectorEqualWithTolerance(const std::vector<uint8_t> &expected,
                                     const std::vector<uint8_t> &actual,
                                     double tolerance)
{
    if (expected.size() != actual.size())
        return false;

    size_t size = expected.size();
    if (size == 0)
        return true;

    if (memcmp(expected.data(), actual.data(), size) == 0)
        return true;

    size_t equal_count = 0;
    size_t not_equal_count = 0;

    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (expected[i] == actual[i])
            ++equal_count;
        else
            ++not_equal_count;
    }

    std::cout << "Vectors are not equal: " << equal_count << " equal, "
              << not_equal_count << " unequal elements." << std::endl;

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
