#pragma once

namespace skeleton_cpp {
namespace calculator {

class Calculator {
public:
    /**
     * @brief 两个数相加
     * @param a 第一个数
     * @param b 第二个数
     * @return 两数之和
     */
    static double add(double a, double b);
    
    /**
     * @brief 两个数相减
     * @param a 被减数
     * @param b 减数
     * @return 两数之差
     */
    static double subtract(double a, double b);
};

} // namespace calculator
} // namespace skeleton_cpp

