#include "skeleton_cpp/calculator/Calculator.hpp"
#include <gtest/gtest.h>

// 测试正数相加
TEST(CalculatorTest, AddPositiveNumbers) {
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::add(2.0, 3.0), 5.0);
}

// 测试负数相加
TEST(CalculatorTest, AddNegativeNumbers) {
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::add(-2.0, -3.0), -5.0);
}

// 测试正数和负数相加
TEST(CalculatorTest, AddPositiveAndNegativeNumbers) {
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::add(5.0, -3.0), 2.0);
}

// 测试小数相加 
TEST(CalculatorTest, AddDecimals) {
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::add(1.5, 2.5), 4.0);
}

// 测试与零相加
TEST(CalculatorTest, AddWithZero) {
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::add(5.0, 0.0), 5.0);
    EXPECT_DOUBLE_EQ(skeleton_cpp::calculator::Calculator::add(0.0, 5.0), 5.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

//