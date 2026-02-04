#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

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
    int cls_id = 0;      // 添加类别ID字段
};

struct SegmentationResult {
    uint8_t *seg_mask = nullptr;
};

struct DetectResultGroup {
    int id = 0;
    int count = 0;
    DetectResult results[64] = {};  // 初始化为默认值
};

struct ResultGroup {
    int id = 0;
    int count = 0;
    std::vector<DetectResult> results; // 检测结果
    std::vector<SegmentationResult> results_seg; // 分割结果
};

} // namespace post_process
} // namespace deploy_percept