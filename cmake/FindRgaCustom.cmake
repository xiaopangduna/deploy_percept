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

# 根据平台设置RGA库路径
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(LIBRGA ${RGA_PATH}/libs/AndroidNdk/${CMAKE_ANDROID_ARCH_ABI}/librga.so)
else()
  # 根据编译器架构选择对应的库文件
  if(CMAKE_C_COMPILER MATCHES "aarch64")
    set(LIB_ARCH aarch64)
  else()
    set(LIB_ARCH armhf)
  endif()

  set(LIBRGA ${RGA_PATH}/libs/Linux/gcc-${LIB_ARCH}/librga.a)
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
    message(WARNING "RGA library not found or not exists")
    message(WARNING "Expected library path: ${LIBRGA}")
endif()

message(STATUS "==============================================================================")