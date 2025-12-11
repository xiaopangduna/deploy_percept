#pragma once

#include <memory>
#include <string>
#include <spdlog/spdlog.h>
// 前向声明，避免包含 spdlog 头文件到公共接口


namespace skeleton_cpp
{
    namespace calculator
    {

        /**
         * @brief 创建应用日志器
         *
         * @param is_debug       true: 仅输出到控制台（带颜色），级别=debug
         *                       false: 控制台 + 文件，级别=info
         * @param log_file_path  生产模式下的日志文件路径（仅 is_debug=false 时生效）
         * @return std::shared_ptr<spdlog::logger>
         */
        std::shared_ptr<spdlog::logger> create_app_logger(
            bool is_debug = false,
            const std::string &log_file_path = "app.log");
    }
} // namespace myapp