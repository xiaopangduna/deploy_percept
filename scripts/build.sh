#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET=""

show_help() {
    echo "deploy_percept 构建脚本"
    echo ""
    echo "用法: bash $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --preset <name>    CMake preset（默认: x86_64-debug）"
    echo "  --help             显示帮助"
    echo ""
    echo "可用 preset:"
    echo "  x86_64-debug                                  x86_64 调试（宿主机）"
    echo "  armv7l-SSC375-release                         Sigmastar SSC375 交叉编译"
    echo "  aarch64-linux-gnu_orange_pi_4_pro_a733-release  Orange Pi 4 Pro A733 交叉编译"
    echo ""
    echo "示例:"
    echo "  bash $0"
    echo "  bash $0 --preset x86_64-debug"
    echo "  bash $0 --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release"
    echo ""
    echo "相关脚本:"
    echo "  bash scripts/test.sh    运行 ctest（需先 build）"
    echo "  bash scripts/install.sh 安装到 install/<platform>/"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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

if [[ -z "${PRESET}" ]]; then
    PRESET="x86_64-debug"
    echo "未指定 preset，使用默认: ${PRESET}"
else
    echo "使用 preset: ${PRESET}"
fi

BUILD_DIR="${PROJECT_ROOT}/build/${PRESET}"

echo "==============================="
echo "配置: cmake --preset=${PRESET}"
echo "构建: cmake --build --preset=${PRESET}"
echo "==============================="

cmake --preset="${PRESET}"
cmake --build --preset="${PRESET}"

echo ""
echo "构建完成: ${BUILD_DIR}"
echo ""
echo "下一步:"
echo "  ctest:   cd ${BUILD_DIR} && ctest --output-on-failure"
echo "  测试:    bash scripts/test.sh --preset ${PRESET}"
echo "  安装:    bash scripts/install.sh --preset ${PRESET}"
