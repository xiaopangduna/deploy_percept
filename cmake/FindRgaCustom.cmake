# FindRgaCustom.cmake - 自定义RGA库查找模块

# git:https://github.com/airockchip/librga/tree/main
# 安装Rga驱动
# 查看Rga驱动版本
# cat /sys/kernel/debug/rkrga/driver_version
# orangepi@orangepi5plus:~/HectorHuang/deploy_percept$ sudo cat /sys/kernel/debug/rkrga/driver_version
# [sudo] password for orangepi: 
# RGA multicore Device Driver: v1.3.1

# 查看rga设备
# orangepi@orangepi5plus:~/HectorHuang/deploy_percept$ ls /dev/rga*
# /dev/rga

# 设置RGA库路径
set(RGA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rga)

# 第一级：根据系统类型判断
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  # Android平台
  set(LIBRGA ${RGA_PATH}/libs/AndroidNdk/${CMAKE_ANDROID_ARCH_ABI}/librga.so)
  
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Linux平台
  # 第二级：根据处理器架构判断
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    # x86_64架构 - RGA不支持
    set(LIBRGA "")
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    # aarch64架构
    set(LIBRGA ${RGA_PATH}/libs/Linux/gcc-${CMAKE_SYSTEM_PROCESSOR}/librga.a)
  else()
    # 其他架构
    message(WARNING "Unsupported processor architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(LIBRGA "")
  endif()
  
else()
  # 其他平台（Windows、macOS等）
  message(WARNING "RGA is not supported on ${CMAKE_SYSTEM_NAME} platform")
  set(LIBRGA "")
endif()

# 设置RGA库的包含目录
set(LIBRGA_INCLUDES ${RGA_PATH}/include)

# 检查RGA库是否存在并设置RGA_FOUND变量
if(EXISTS ${LIBRGA})
    set(RGA_FOUND TRUE)
    message(STATUS "RGA library found successfully")
    message(STATUS "RGA path: ${RGA_PATH}")
    message(STATUS "RGA library: ${LIBRGA}")
    message(STATUS "RGA includes: ${LIBRGA_INCLUDES}")
else()
    set(RGA_FOUND FALSE)
    
    message(STATUS "WARNING RGA library not found !!!")

endif()

message(STATUS "==============================================================================")