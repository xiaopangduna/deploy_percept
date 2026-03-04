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

struct DetectionObject {
    char name[16] = {};  // 初始化字符数组为全零
    BoxRect box{};
    float prop = 0.0f;
    int cls_id = 0;      // 添加类别ID字段
};

// 移除SegmentationResult结构体，直接使用std::vector<uint8_t>
// 移除DetectResultGroup结构体，统一使用ResultGroup

struct ResultGroup {
    int id = 0;
    int count = 0;
    std::vector<DetectionObject> results; // 检测结果
    std::vector<uint8_t> segmentation_masks; // 分割掩码，对于检测场景为空
};

} // namespace post_process
} // namespace deploy_percept