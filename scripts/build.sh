#!/bin/bash

# 显示帮助信息
show_help() {
    echo "skeleton_cpp 项目构建脚本"
    echo ""
    echo "用法:"
    echo " bash $0 [选项]"
    echo ""
    echo "可选选项:"
    echo "  --preset <预设名>          使用指定的构建预设 (默认: x86_64-debug)"
    echo "                            支持的预设有:"
    echo "                              - x86_64-debug: x86_64架构调试版本"
    echo "                              - aarch64-release: aarch64架构发布版本"
    echo "  --help                    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 使用默认预设 (x86_64-debug)"
    echo "  bash $0"
    echo ""
    echo "  # 使用指定预设"
    echo "  bash $0 --preset x86_64-debug"
    echo "  bash $0 --preset aarch64-release"
    echo ""
    echo "注意:"
    echo "  此脚本使用CMakePresets.json中的预设配置进行项目构建"
}

# 初始化变量
PRESET=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 如果没有指定预设，则使用默认值
if [ -z "$PRESET" ]; then
    PRESET="x86_64-debug"
    echo "未指定预设，使用默认: $PRESET"
else
    echo "使用指定预设: $PRESET"
fi

echo "使用CMakePresets.json构建项目"
echo "==============================="

echo "执行构建命令: cmake --preset=$PRESET && cmake --build --preset=$PRESET"
echo ""

# 使用指定的预设配置项目
cmake --preset=$PRESET

# 构建项目
cmake --build --preset=$PRESET

echo ""
echo "构建完成！"
echo "==============================="
echo "可用预设："
echo "  x86_64-debug   - x86_64架构调试版本"
echo "  aarch64-release - aarch64架构发布版本"