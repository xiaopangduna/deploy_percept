#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace deploy_percept
{

/** 非 owning 内存视图：统一 Mapped / HostCopy 等路径下的可读 tensor 缓冲 */
enum class TensorDtype : std::uint8_t
{
    FP32,
    INT8,
};

struct TensorView
{
    const void *data{nullptr};
    std::size_t byte_size{0};
    TensorDtype dtype{TensorDtype::FP32};
};

/** 轴对齐框，像素整数坐标（与 OpenCV 绘制一致） */
struct BoxRect
{
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;
};

/** 单个关键点：x/y 为像素坐标，score 为置信度 */
struct Keypoint
{
    float x = 0.f;
    float y = 0.f;
    float score = 0.f;
};

inline constexpr std::size_t kCocoPoseKeypointCount = 17;

// ---------------------------------------------------------------------------
// 检测（Detect）
// ---------------------------------------------------------------------------

struct Detection
{
    BoxRect box{};
    int class_id = 0;
    float score = 0.f;
};

struct DetectionResult
{
    std::vector<Detection> detections;
};

// ---------------------------------------------------------------------------
// 分割（Segmentation）：检测框 + 全图 label mask
// ---------------------------------------------------------------------------

struct SegmentationResult
{
    std::vector<Detection> detections;

    /**
     * 与模型输入同分辨率的 label mask，长度 = width * height。
     * 0 = 背景；非 0 = 该像素 class_id + 1（与 mergeBoxMasks 编码一致）。
     */
    std::vector<std::uint8_t> label_mask;
};

// ---------------------------------------------------------------------------
// 姿态（Pose）
// ---------------------------------------------------------------------------

struct PoseDetection
{
    BoxRect box{};
    int class_id = 0;
    float score = 0.f;
    std::array<Keypoint, kCocoPoseKeypointCount> keypoints{};
};

struct PoseResult
{
    std::vector<PoseDetection> detections;
};

// ---------------------------------------------------------------------------
// 分类（Classification）
// ---------------------------------------------------------------------------

struct ClassificationResult
{
    int class_id = -1;
    float score = 0.f;
};

} // namespace deploy_percept
