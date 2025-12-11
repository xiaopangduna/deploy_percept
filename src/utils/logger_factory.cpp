// src/utils/logger_factory.cpp
#include "skeleton_cpp/utils/logger_factory.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <memory>

namespace skeleton_cpp::calculator {

std::shared_ptr<spdlog::logger> create_app_logger(
    bool is_debug,
    const std::string& log_file_path)
{
    const char* pattern = is_debug
        ? "[%H:%M:%S.%e] [%^%l%$] %v"
        : "[%Y-%m-%d %H:%M:%S] [%l] %v";

    if (is_debug) {
        // 调试模式：彩色控制台
        auto logger = spdlog::stdout_color_mt("app");
        logger->set_pattern(pattern);
        logger->set_level(spdlog::level::debug);
        return logger;
    } else {
        // 生产模式：控制台 + 文件
        // 方法：分别创建两个 logger，然后组合它们的 sinks
        auto console = spdlog::stdout_color_mt("console_temp");
        auto file = spdlog::basic_logger_mt("file_temp", log_file_path);

        // 提取 sinks（注意：必须从已存在的 logger 中获取）
        auto sinks = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sinks->set_pattern(pattern);

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path, true);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] [%n] %v");

        // 创建组合 logger
        auto logger = std::make_shared<spdlog::logger>("app", spdlog::sinks_init_list{{
            sinks,
            file_sink
        }});

        logger->set_level(spdlog::level::info);
        spdlog::register_logger(logger);
        return logger;
    }
}

} // namespace skeleton_cpp::calculator