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

bool CompareSegmentationMaskVectors(const std::vector<uint8_t> &expected,
                                    const std::vector<uint8_t> &actual)
{
    bool match = true;

    ::testing::ScopedTrace trace(__FILE__, __LINE__,
                                 "Comparing segmentation masks");

    EXPECT_EQ(expected.size(), actual.size());
    if (expected.size() != actual.size())
    {
        match = false;
        return match;
    }

    // 逐字节比较
    for (size_t j = 0; j < expected.size(); ++j)
    {
        SCOPED_TRACE("Mask byte index " + std::to_string(j));
        EXPECT_EQ(expected[j], actual[j]);
        if (::testing::Test::HasFailure())
        {
            match = false;
        }
    }

    return match;
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