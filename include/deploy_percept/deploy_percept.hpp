#pragma once


// 主要的库头文件，包含所有公共接口

// 包含预处理模块
#include "pre_process/image_pre_process.hpp"

// 包含后处理模块
#include "post_process/PostProcessFactory.hpp"
#include "post_process/YoloV5DetectPostProcess.hpp"
#include "post_process/YoloV5DetectPostProcessAwnn.hpp"
#include "post_process/YoloV8DetectPostProcessAwnn.hpp"
#include "post_process/YoloV8PosePostProcess.hpp"
#include "post_process/types.hpp"

// 包含引擎模块
#include "engine/EngineFactory.hpp"
#ifdef AWNN_FOUND
#include "engine/AwnnEngine.hpp"
#include "engine/VipLiteRuntime.hpp"
#include "engine/AwnnResultGuard.hpp"
#endif
#include "engine/RknnEngine.hpp"


// 包含工具模块
#include "utils/logger_factory.hpp"