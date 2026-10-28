#pragma once

#include <string>

namespace sensor_preview
{

/** 检测支路配置。模型留在板上，这里只传路径。 */
struct DetectConfig
{
    std::string model_path;
    /** 与预览 SCL 相同的旋转角度：0 / 90 / 180 / 270。框要转到屏幕坐标。 */
    int rotate_degree = 90;
};

/**
 * ISP 检测支路已经在输出 NV12 之后启动。
 * 本对象只负责 IPU、后处理和 /dev/fb0 上的框。
 */
class DetectSession
{
public:
    DetectSession() = default;
    ~DetectSession();

    DetectSession(const DetectSession &) = delete;
    DetectSession &operator=(const DetectSession &) = delete;

    bool start(const DetectConfig &config);
    void stop();

private:
    struct Impl;
    Impl *impl_ = nullptr;
};
}  // namespace sensor_preview
