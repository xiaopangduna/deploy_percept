# aarch64-linux-gnu toolchain

# 查看编译器是否安装
# aarch64-linux-gnu-gcc --version
# sudo apt install -y \
#   gcc-aarch64-linux-gnu \
#   g++-aarch64-linux-gnu \
#   binutils-aarch64-linux-gnu
# xiaopangdun@lovelyboy:~/project/deploy_percept$ aarch64-linux-gnu-gcc -dumpmachine
# aarch64-linux-gnu
# 验证可执行文件是否为aarch64
# file build/aarch64-release/tests/test_YoloV5DetectPostProcess

message(STATUS "=== Using aarch64 cross compiler ===")

# 目标系统
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 编译器（不要乱 FORCE，除非你非常清楚后果）
set(CMAKE_C_COMPILER   /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

# Root path
set(CMAKE_FIND_ROOT_PATH
    /usr/aarch64-linux-gnu
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# 查找策略（推荐）
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

message(STATUS "CMAKE_SYSTEM_NAME       = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR  = ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_C_COMPILER        = ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER      = ${CMAKE_CXX_COMPILER}")