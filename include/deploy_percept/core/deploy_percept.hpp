#pragma once

// core 库入口：engine + post_process
// 链接 deploy_percept_core 后使用

#include "deploy_percept/common/types.hpp"

#include "deploy_percept/core/post_process/YoloDetectPostProcess.hpp"
#include "deploy_percept/core/post_process/YoloV5DetectPostProcessRknn.hpp"
#include "deploy_percept/core/post_process/YoloV5SegPostProcess.hpp"
#include "deploy_percept/core/post_process/YoloV8SegPostProcess.hpp"
#ifdef AWNN_FOUND
#include "deploy_percept/core/post_process/YoloV5DetectPostProcessAwnn.hpp"
#include "deploy_percept/core/post_process/YoloV8DetectPostProcessAwnn.hpp"
#endif
#include "deploy_percept/core/post_process/ClassificationPostProcessAwnn.hpp"
#include "deploy_percept/core/post_process/YoloV8PosePostProcess.hpp"

#ifdef AWNN_FOUND
#include "deploy_percept/core/engine/AwnnEngine.hpp"
#include "deploy_percept/core/engine/EngineResultGuard.hpp"
#endif
#include "deploy_percept/core/engine/RknnEngine.hpp"
