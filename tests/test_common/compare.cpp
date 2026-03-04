#include "compare.hpp"
#include <gtest/gtest.h>
#include <cstring>
#include <cmath>

bool CompareSegmentationMaskVectors(const std::vector<uint8_t>& expected,
                                   const std::vector<uint8_t>& actual) {
    bool match = true;
    
    ::testing::ScopedTrace trace(__FILE__, __LINE__, 
        "Comparing segmentation masks");
        
    EXPECT_EQ(expected.size(), actual.size());
    if (expected.size() != actual.size()) {
        match = false;
        return match;
    }
    
    // 逐字节比较
    for (size_t j = 0; j < expected.size(); ++j) {
        SCOPED_TRACE("Mask byte index " + std::to_string(j));
        EXPECT_EQ(expected[j], actual[j]);
        if (::testing::Test::HasFailure()) {
            match = false;
        }
    }
    
    return match;
}

bool CompareDetectResultVectors(const std::vector<deploy_percept::post_process::DetectionObject>& expected,
                               const std::vector<deploy_percept::post_process::DetectionObject>& actual) {
    bool match = true;
    
    ::testing::ScopedTrace trace(__FILE__, __LINE__, 
        "Comparing detection results");
    
    // 不检查大小，因为可能有额外的低置信度检测结果
    
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto& exp = expected[i];
        const auto& act = actual[i];
        
        // 使用 SCOPED_TRACE 帮助定位哪个元素失败
        ::testing::ScopedTrace trace_elem(__FILE__, __LINE__,
                                         "Comparing element " + std::to_string(i));
        
        if (exp.cls_id != act.cls_id) {
            EXPECT_EQ(exp.cls_id, act.cls_id);
            match = false;
        }
        
        if (std::strcmp(exp.name, act.name) != 0) {
            EXPECT_STREQ(exp.name, act.name);
            match = false;
        }
        
        if (std::abs(exp.prop - act.prop) > 1e-4) {
            EXPECT_NEAR(exp.prop, act.prop, 1e-4);
            match = false;
        }
        
        if (exp.box.left != act.box.left ||
            exp.box.top != act.box.top ||
            exp.box.right != act.box.right ||
            exp.box.bottom != act.box.bottom) {
            EXPECT_EQ(exp.box.left, act.box.left);
            EXPECT_EQ(exp.box.top, act.box.top);
            EXPECT_EQ(exp.box.right, act.box.right);
            EXPECT_EQ(exp.box.bottom, act.box.bottom);
            match = false;
        }
    }
    
    return match;
}