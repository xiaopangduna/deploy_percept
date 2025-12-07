# x86_64 工具链配置文件
# 用于明确指定x86_64平台构建

# 设置目标系统
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# 指定编译器
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
