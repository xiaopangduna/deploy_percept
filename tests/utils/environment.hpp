#pragma once

#include <gtest/gtest.h>
#include "skeleton_cpp/utils/logger_factory.hpp"

// 全局环境类，用于管理整个测试过程中的日志系统
class GlobalLoggerEnvironment : public ::testing::Environment
{
public:
    static std::shared_ptr<spdlog::logger> logger;

    void SetUp() override
    {
        // 初始化日志系统
        try
        {
            logger = skeleton_cpp::calculator::create_app_logger(true);
        }
        catch (...)
        {
            // 如果日志已经存在，则获取现有日志实例
            logger = spdlog::get("app");
        }
    }

    void TearDown() override
    {
        if (logger)
        {
            logger.reset();
        }
    }
};

// 静态成员变量定义
std::shared_ptr<spdlog::logger> GlobalLoggerEnvironment::logger = nullptr;