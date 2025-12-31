# rknn runtime
set(RKNN_PLATFORM "rk3588")

# 第一级：检查平台支持
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Linux平台
  # 第二级：检查架构支持
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    # x86_64架构不支持
    set(LIBRKNNRT "")
  else()
    # ARM架构，继续正常处理
    # for rknpu2
    if (RKNN_PLATFORM STREQUAL "rk3588" OR RKNN_PLATFORM STREQUAL "rk3576" OR RKNN_PLATFORM STREQUAL "rk356x" OR RKNN_PLATFORM STREQUAL "rv1106" OR RKNN_PLATFORM STREQUAL "rv1103" OR RKNN_PLATFORM STREQUAL "rv1126b")
      set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknpu2)
      if (RKNN_PLATFORM STREQUAL "rk3588" OR RKNN_PLATFORM STREQUAL "rk356x" OR RKNN_PLATFORM STREQUAL "rk3576")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/librknnrt.so)
      endif()
      if (RKNN_PLATFORM STREQUAL "rv1126b")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/librknnrt.so)
      endif()
      if (RKNN_PLATFORM STREQUAL "rv1106" OR RKNN_PLATFORM STREQUAL "rv1103")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/armhf-uclibc/librknnmrt.so)
      endif()
      # 使用生成器表达式避免相对路径问题
      set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)
    endif()
    
    # for rknpu1
    if(RKNN_PLATFORM STREQUAL "rk1808" OR RKNN_PLATFORM STREQUAL "rv1109" OR RKNN_PLATFORM STREQUAL "rv1126")
      set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknpu1)
      set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/librknn_api.so)
      # 使用生成器表达式避免相对路径问题
      set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)
    endif()
    
    set(LIBRKNNRT ${LIBRKNNRT})
  endif()
  
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
  # Android平台
  # for rknpu2
  if (RKNN_PLATFORM STREQUAL "rk3588" OR RKNN_PLATFORM STREQUAL "rk3576" OR RKNN_PLATFORM STREQUAL "rk356x" OR RKNN_PLATFORM STREQUAL "rv1106" OR RKNN_PLATFORM STREQUAL "rv1103" OR RKNN_PLATFORM STREQUAL "rv1126b")
    set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknpu2)
    # Android通常使用Android目录，可能根据ABI选择
    if(CMAKE_ANDROID_ARCH_ABI)
      set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
    else()
      set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/librknnrt.so)
    endif()
    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)
  endif()
  
  # for rknpu1
  if(RKNN_PLATFORM STREQUAL "rk1808" OR RKNN_PLATFORM STREQUAL "rv1109" OR RKNN_PLATFORM STREQUAL "rv1126")
    set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknpu1)
    if(CMAKE_ANDROID_ARCH_ABI)
      set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_ANDROID_ARCH_ABI}/librknn_api.so)
    else()
      set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/librknn_api.so)
    endif()
    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)
  endif()
  
else()
  # 其他平台（Windows、macOS等）
  message(WARNING "RKNN runtime is not supported on ${CMAKE_SYSTEM_NAME} platform")
  set(LIBRKNNRT "")
endif()

message(STATUS "==============================================================================")

# 检查RKNN库是否存在并设置RKNN_FOUND变量
if(LIBRKNNRT AND EXISTS ${LIBRKNNRT})
    set(RKNN_FOUND TRUE)
    message(STATUS "RKNN runtime found successfully")
    message(STATUS "RKNN Platform: ${RKNN_PLATFORM}")
    message(STATUS "RKNN path: ${RKNN_PATH}")
    message(STATUS "RKNN library: ${LIBRKNNRT}")
    message(STATUS "RKNN includes: ${LIBRKNNRT_INCLUDES}")
else()
    set(RKNN_FOUND FALSE)
    # 库路径被设置但文件不存在
    message(STATUS "WARNING RKNN runtime not found !!!")

endif()

message(STATUS "==============================================================================")