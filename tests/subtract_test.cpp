#include "skeleton_cpp/calculator/Calculator.hpp"
#include <gtest/gtest.h>

// 测试正数相减
TEST(SubtractTest, SubtractPositiveNumbers) {
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::subtract(5.0, 3.0), 2.0);
}

// 测试负数相减
TEST(SubtractTest, SubtractNegativeNumbers) {
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::subtract(-5.0, -3.0), -2.0);
}

// 测试正数减负数
TEST(SubtractTest, SubtractPositiveAndNegativeNumbers) {
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::subtract(5.0, -3.0), 8.0);
}

// 测试小数相减
TEST(SubtractTest, SubtractDecimals) {
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::subtract(4.5, 2.5), 2.0);
}

// 测试减零
TEST(SubtractTest, SubtractZero) {
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::subtract(5.0, 0.0), 5.0);
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::subtract(0.0, 5.0), -5.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}