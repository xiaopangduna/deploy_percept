#pragma once

#include <cstdint>
#include <cstring>

namespace deploy_percept {
namespace post_process {

struct BoxRect {
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;
};

struct DetectResult {
    char name[16] = {};  // 初始化字符数组为全零
    BoxRect box{};
    float prop = 0.0f;
};

struct DetectResultGroup {
    int id = 0;
    int count = 0;
    DetectResult results[64] = {};  // 初始化为默认值
};

} // namespace post_process
} // namespace deploy_percept

