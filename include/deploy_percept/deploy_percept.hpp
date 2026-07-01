#pragma once

// core 库入口：engine + post_process + types
// 链接 deploy_percept_core（或 deploy_percept 别名）后使用

#include "types.hpp"

#include "post_process/PostProcessFactory.hpp"
#include "post_process/YoloV5DetectPostProcess.hpp"
#ifdef AWNN_FOUND
#include "post_process/YoloV5DetectPostProcessAwnn.hpp"
#include "post_process/YoloV8DetectPostProcessAwnn.hpp"
#endif
#include "post_process/YoloV8PosePostProcess.hpp"

#include "engine/EngineFactory.hpp"
#ifdef AWNN_FOUND
#include "engine/AwnnEngine.hpp"
#include "engine/VipLiteRuntime.hpp"
#include "engine/AwnnResultGuard.hpp"
#endif
#include "engine/RknnEngine.hpp"