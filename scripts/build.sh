#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET=""
DO_INSTALL=0

show_help() {
    echo "deploy_percept 构建脚本"
    echo ""
    echo "用法: bash $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --preset <name>    CMake preset（默认: x86_64-debug）"
    echo "  --install          构建完成后执行 cmake --install"
    echo "  --help             显示帮助"
    echo ""
    echo "可用 preset 见 CMakePresets.json，或: cmake --list-presets"
    echo ""
    echo "示例:"
    echo "  bash $0"
    echo "  bash $0 --preset x86_64-debug"
    echo "  bash $0 --preset x86_64-debug --install"
    echo ""
    echo "相关脚本:"
    echo "  bash scripts/test.sh         build + install + 跑测"
    echo "  bash scripts/sync_install_to_board.sh 仅同步 install 到开发板"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --install)
            DO_INSTALL=1
            shift
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

# CMakePresets.json version 3 需要 CMake >= 3.21；Ubuntu 20.04 自带 3.16 不支持 --preset
CMAKE_VER="$(cmake --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
CMAKE_MAJOR="${CMAKE_VER%%.*}"
CMAKE_MINOR="${CMAKE_VER#*.}"
if [[ -z "${CMAKE_VER}" ]] \
   || (( CMAKE_MAJOR < 3 )) \
   || (( CMAKE_MAJOR == 3 && CMAKE_MINOR < 21 )); then
    echo "错误: 需要 CMake >= 3.21 以支持 --preset（当前: $(cmake --version 2>/dev/null | head -1 || echo '未找到 cmake')）" >&2
    echo "  Sigmastar 容器请 Rebuild（.devcontainer/ubuntu-20.04-sigmastar 已从 Kitware 安装新版 CMake）。" >&2
    exit 1
fi

echo "==============================="
echo "配置: cmake --preset=${PRESET}"
echo "构建: cmake --build --preset=${PRESET}"
echo "==============================="

##########################################################################
# 核心构建逻辑
cmake --preset="${PRESET}"
# clangd 从工程根目录读 compile_commands.json（.gitignore，不入库）
ln -sfn "build/${PRESET}/compile_commands.json" "${PROJECT_ROOT}/compile_commands.json"
cmake --build --preset="${PRESET}"


echo ""
echo "构建完成: ${BUILD_DIR}"

if [[ "${DO_INSTALL}" -eq 1 ]]; then
    if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        echo "错误: 构建目录未配置: ${BUILD_DIR}" >&2
        exit 1
    fi
    INSTALL_DIR="$(grep -E '^CMAKE_INSTALL_PREFIX:' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2)"
    echo ""
    echo "=== cmake --install ==="
    ##########################################################################
    # 安装逻辑
    cmake --install "${BUILD_DIR}"
    echo "安装完成: ${INSTALL_DIR}"
    exit 0
fi

echo ""
echo "下一步: bash scripts/build.sh --preset ${PRESET} --install"
echo "        bash scripts/test.sh --preset ${PRESET}"
