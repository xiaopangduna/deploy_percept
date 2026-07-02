#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
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

struct BoxRect
{
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;
};

struct DetectionObject
{
    float prop = 0.0f;
    int cls_id = 0;
    char name[16] = {};
    BoxRect box{};
};

struct ResultGroup
{
    int id = 0;
    int count = 0;
    std::vector<DetectionObject> detection_objects;
    std::vector<std::uint8_t> segmentation_mask;
};

/** YOLOv8-Pose：检测框 + 17 个 COCO 关键点 (x, y, conf)，坐标为原图像素 */
struct PoseDetectionObject
{
    BoxRect box{};
    float prop = 0.f;
    int cls_id = 0;
    float keypoints[17][3] = {};
};

struct PoseResultGroup
{
    int count = 0;
    std::vector<PoseDetectionObject> objects;
};

struct ClassificationEntry
{
    int class_id = 0;
    float prob = 0.f;
    std::string label;
};

struct ClassificationResult
{
    bool success = false;
    std::string message;
    std::vector<ClassificationEntry> top_k;
};

namespace post_process
{

using TensorDtype = deploy_percept::TensorDtype;
using TensorView = deploy_percept::TensorView;
using BoxRect = deploy_percept::BoxRect;
using DetectionObject = deploy_percept::DetectionObject;
using ResultGroup = deploy_percept::ResultGroup;
using PoseDetectionObject = deploy_percept::PoseDetectionObject;
using PoseResultGroup = deploy_percept::PoseResultGroup;
using ClassificationEntry = deploy_percept::ClassificationEntry;
using ClassificationResult = deploy_percept::ClassificationResult;

} // namespace post_process

} // namespace deploy_percept
