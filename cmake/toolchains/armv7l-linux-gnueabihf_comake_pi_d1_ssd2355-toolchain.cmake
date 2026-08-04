# CoMake Pi D1 (Sigmastar SSD2355) arm-unknown-linux-gnueabihf toolchain (gcc 12.4 / glibc 2.37)
# devcontainer bind mount: 宿主机 /opt/toolchains -> 容器 /opt/toolchains
#
# TOOLCHAIN_ROOT / TOOLCHAIN_PREFIX 为本平台工具链的唯一配置来源；
# shell 构建脚本（如 builder_opencv.sh）从此文件解析，请勿在其他脚本重复硬编码路径。

set(TOOLCHAIN_ROOT "/opt/toolchains/arm-unknown-linux-gcc-12.4.0-glibc-2.37-gnueabihf" CACHE PATH "CoMake Pi D1 SSD2355 toolchain root")
set(TOOLCHAIN_PREFIX "arm-unknown-linux-gnueabihf")
set(TARGET_TRIPLE "arm-unknown-linux-gnueabihf")

# third_party/<tag>/<lib>/ 目录名，与 builder --platform 一致
set(THIRD_PARTY_PLATFORM_TAG "armv7l-linux-gnueabihf_comake_pi_d1_ssd2355" CACHE STRING "Third-party install platform tag")
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/../../third_party" CACHE PATH "Third-party root")
set(THIRD_PARTY_PLATFORM_DIR "${THIRD_PARTY_DIR}/${THIRD_PARTY_PLATFORM_TAG}" CACHE PATH "Platform third-party install dir")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

set(CMAKE_C_COMPILER "${TOOLCHAIN_ROOT}/bin/${TOOLCHAIN_PREFIX}-gcc" CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_ROOT}/bin/${TOOLCHAIN_PREFIX}-g++" CACHE FILEPATH "C++ compiler" FORCE)
set(CMAKE_SYSROOT "${TOOLCHAIN_ROOT}/${TARGET_TRIPLE}/sysroot" CACHE PATH "Sysroot" FORCE)

set(CMAKE_FIND_ROOT_PATH
    ${CMAKE_SYSROOT}
    ${THIRD_PARTY_PLATFORM_DIR}
)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# 与工具链默认目标一致；显式写入 compile_commands 供 clangd 正确解析 ARM 头文件
add_compile_options(
    -mfloat-abi=hard
    -mfpu=neon-vfpv4
    -mthumb
    -march=armv7-a+neon-vfpv4
)

message(STATUS "==============================================================================")
message(STATUS "armv7l-linux-gnueabihf_comake_pi_d1_ssd2355 toolchain:")
message(STATUS "  CMAKE_TOOLCHAIN_FILE     = ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "  TOOLCHAIN_ROOT           = ${TOOLCHAIN_ROOT}")
message(STATUS "  TOOLCHAIN_PREFIX         = ${TOOLCHAIN_PREFIX}")
message(STATUS "  TARGET_TRIPLE            = ${TARGET_TRIPLE}")
message(STATUS "  THIRD_PARTY_PLATFORM_TAG = ${THIRD_PARTY_PLATFORM_TAG}")
message(STATUS "  THIRD_PARTY_PLATFORM_DIR = ${THIRD_PARTY_PLATFORM_DIR}")
message(STATUS "  CMAKE_SYSTEM_NAME        = ${CMAKE_SYSTEM_NAME}")
message(STATUS "  CMAKE_SYSTEM_PROCESSOR   = ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  CMAKE_C_COMPILER         = ${CMAKE_C_COMPILER}")
message(STATUS "  CMAKE_CXX_COMPILER       = ${CMAKE_CXX_COMPILER}")
message(STATUS "  CMAKE_SYSROOT            = ${CMAKE_SYSROOT}")
message(STATUS "  CMAKE_FIND_ROOT_PATH     = ${CMAKE_FIND_ROOT_PATH}")
message(STATUS "  FIND_ROOT_PATH_MODE      = PROGRAM:NEVER LIBRARY:ONLY INCLUDE:ONLY PACKAGE:ONLY")
message(STATUS "  COMPILE_OPTIONS          = -mfloat-abi=hard -mfpu=neon-vfpv4 -mthumb -march=armv7-a+neon-vfpv4")
message(STATUS "==============================================================================")
