#include "compare.hpp"
#include <gtest/gtest.h>

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