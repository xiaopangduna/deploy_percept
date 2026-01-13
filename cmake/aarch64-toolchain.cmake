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

# 编译器（必须是 CACHE + FILEPATH）
set(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc" CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++" CACHE FILEPATH "C++ compiler" FORCE)

# 目标根路径（Ubuntu multiarch）和项目第三方库路径
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# 查找策略（关键）
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")