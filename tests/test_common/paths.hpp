#pragma once

#include <filesystem>
#include <string>

namespace percept {
namespace test {

// 数据根：build tree 为项目根；install 后为 <prefix>/share/percept
std::filesystem::path resolve_root();

// apps/<relative>，例如 yolov5_seg_rknn/bus.jpg
std::filesystem::path app_data(const std::string &relative);

// 测试输出目录
std::filesystem::path output_dir();

} // namespace test
} // namespace percept
