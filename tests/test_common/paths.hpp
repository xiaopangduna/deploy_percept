#pragma once

#include "deploy_percept/utils/fixture_path.hpp"

#include <filesystem>
#include <string>

namespace percept {
namespace test {

// tests/fixtures/<relative>（install 后为 share/percept/fixtures/<relative>）
inline std::filesystem::path fixture_data(const std::string &relative)
{
    return deploy_percept::utils::fixture_path(relative);
}

// 测试输出目录（ctest: tmp/；install: var/percept/output）
std::filesystem::path output_dir();

} // namespace test
} // namespace percept
