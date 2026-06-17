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

set(THIRD_PARTY_PLATFORM_TAG "aarch64" CACHE STRING "Third-party install platform tag")
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/../../third_party" CACHE PATH "Third-party root")
set(THIRD_PARTY_PLATFORM_DIR "${THIRD_PARTY_DIR}/${THIRD_PARTY_PLATFORM_TAG}" CACHE PATH "Platform third-party install dir")

# 编译器（必须是 CACHE + FILEPATH）
set(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc" CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++" CACHE FILEPATH "C++ compiler" FORCE)

# 目标根路径（Ubuntu multiarch + 自编译第三方库）
set(CMAKE_FIND_ROOT_PATH
    /usr/aarch64-linux-gnu
    ${THIRD_PARTY_PLATFORM_DIR}
)

# 查找策略（关键）
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
