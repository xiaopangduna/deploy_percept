#pragma once

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

namespace image
{

/** 像素格式（通道顺序与常见 OpenCV imread 默认 BGR 一致） */
enum class PixelFormat : std::uint8_t
{
    BGR888,
    RGB888,
    GRAY8,
};

/** 只读、不拥有内存的图像视图（语义类似 cv::Mat 指向外部 buffer） */
struct ImageView
{
    const std::uint8_t *data{nullptr};
    int width{0};
    int height{0};
    /** 每行字节数；0 表示紧凑布局 width * channels() */
    int stride{0};
    PixelFormat format{PixelFormat::BGR888};

    int channels() const;
    std::size_t row_bytes() const;
    std::size_t byte_size() const;
    bool empty() const;
};

/** 可写图像视图；不负责分配/释放 */
struct ImageMut
{
    std::uint8_t *data{nullptr};
    int width{0};
    int height{0};
    int stride{0};
    PixelFormat format{PixelFormat::BGR888};

    ImageView view() const;
    int channels() const;
    std::size_t row_bytes() const;
    std::size_t byte_size() const;
    bool empty() const;
};

int channels(PixelFormat format);

} // namespace image

namespace post_process
{

using TensorDtype = deploy_percept::TensorDtype;
using TensorView = deploy_percept::TensorView;
using BoxRect = deploy_percept::BoxRect;
using DetectionObject = deploy_percept::DetectionObject;
using ResultGroup = deploy_percept::ResultGroup;
using PoseDetectionObject = deploy_percept::PoseDetectionObject;
using PoseResultGroup = deploy_percept::PoseResultGroup;

} // namespace post_process

} // namespace deploy_percept
