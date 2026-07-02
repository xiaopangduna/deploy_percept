#pragma once

// core 库入口：engine + post_process
// 链接 deploy_percept_core 后使用

#include "deploy_percept/common/types.hpp"

#include "deploy_percept/core/post_process/PostProcessFactory.hpp"
#include "deploy_percept/core/post_process/YoloV5DetectPostProcess.hpp"
#include "deploy_percept/core/post_process/YoloV5SegPostProcess.hpp"
#include "deploy_percept/core/post_process/YoloV8SegPostProcess.hpp"
#ifdef AWNN_FOUND
#include "deploy_percept/core/post_process/YoloV5DetectPostProcessAwnn.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessAwnn.hpp"
#endif
#include "deploy_percept/core/post_process/YoloV8PosePostProcess.hpp"

#include "deploy_percept/core/engine/EngineFactory.hpp"
#ifdef AWNN_FOUND
#include "deploy_percept/core/engine/AwnnEngine.hpp"
#include "deploy_percept/core/engine/VipLiteRuntime.hpp"
#include "deploy_percept/core/engine/AwnnResultGuard.hpp"
#endif
#include "deploy_percept/core/engine/RknnEngine.hpp"
