#pragma once

#include "deploy_percept/utils/fixture_path.hpp"

#include <filesystem>
#include <string>

namespace percept {
namespace test {

// 数据根：install 后为 <prefix>/share/percept；ctest 时 PERCEPT_ROOT 为项目根
std::filesystem::path resolve_root();

// tests/fixtures/<relative>（install 后为 share/percept/fixtures/<relative>）
inline std::filesystem::path fixture_data(const std::string &relative)
{
    return deploy_percept::utils::fixture_path(relative);
}

// 测试输出目录
std::filesystem::path output_dir();

} // namespace test
} // namespace percept
