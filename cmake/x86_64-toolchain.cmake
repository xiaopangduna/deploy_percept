# x86_64 toolchain file
message(STATUS "=== Using x86_64 compiler toolchain ===")

# 目标系统
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(THIRD_PARTY_PLATFORM_TAG "x86_64" CACHE STRING "Third-party install platform tag")
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party" CACHE PATH "Third-party root")
set(THIRD_PARTY_PLATFORM_DIR "${THIRD_PARTY_DIR}/${THIRD_PARTY_PLATFORM_TAG}" CACHE PATH "Platform third-party install dir")

# 显式指定完整路径 + 写入 CACHE
set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE FILEPATH "C++ compiler" FORCE)
