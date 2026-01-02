message(STATUS "=== 正在使用aarch64交叉编译器 ===")
# 查看编译器是否安装
# aarch64-linux-gnu-gcc --version
# sudo apt install -y \
#   gcc-aarch64-linux-gnu \
#   g++-aarch64-linux-gnu \
#   binutils-aarch64-linux-gnu
# xiaopangdun@lovelyboy:~/project/deploy_percept$ aarch64-linux-gnu-gcc -dumpmachine
# aarch64-linux-gnu
# file build/aarch64-release/tests/test_YoloV5DetectPostProcess

# 设置目标系统
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 指定交叉编译器
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# ==================== 新增的核心配置 ====================
# 1. 设置目标系统的根目录（Sysroot）
#    这是交叉编译的“基石”，CMake会在这里查找目标平台的头文件和库。
#    请根据您的实际情况设置路径，通常由交叉编译器包提供。
#    可以通过命令 `aarch64-linux-gnu-gcc -print-sysroot` 查看默认值。
set(CMAKE_SYSROOT "")
#    将 sysroot 也添加到查找根路径中
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

# 2. 调整CMake的查找策略（这是最关键的一步）
#    强制CMake在目标环境中查找库和头文件，而不是主机环境。
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)  # 可执行程序仍在主机PATH中查找
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)   # 库文件 ONLY 表示仅在目标目录查找
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)   # 头文件 ONLY 表示仅在目标目录查找
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)   # CMake包 ONLY 表示仅在目标目录查找

# 3. （可选）设置编译器标志，确保生成正确的目标代码
add_compile_options(-march=armv8-a)

set(CMAKE_EXE_LINKER_FLAGS "-L/usr/aarch64-linux-gnu/lib -Wl,-rpath-link,/usr/aarch64-linux-gnu/lib")
set(CMAKE_SHARED_LINKER_FLAGS "-L/usr/aarch64-linux-gnu/lib -Wl,-rpath-link,/usr/aarch64-linux-gnu/lib")
set(CMAKE_MODULE_LINKER_FLAGS "-L/usr/aarch64-linux-gnu/lib -Wl,-rpath-link,/usr/aarch64-linux-gnu/lib")

message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")