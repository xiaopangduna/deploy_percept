#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET="x86_64-debug"

show_help() {
    echo "deploy_percept 安装脚本"
    echo ""
    echo "用法: bash $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --preset <name>    CMake preset（默认: x86_64-debug）"
    echo "  --help             显示帮助"
    echo ""
    echo "可用 preset:"
    echo "  x86_64-debug"
    echo "  armv7l-SSC375-release"
    echo "  aarch64-linux-gnu_orange_pi_4_pro_a733-release"
    echo ""
    echo "示例:"
    echo "  bash scripts/build.sh --preset x86_64-debug"
    echo "  bash $0 --preset x86_64-debug"
    echo ""
    echo "相关脚本:"
    echo "  bash scripts/build.sh   编译"
    echo "  bash scripts/test.sh    测试（build tree / install tree / 开发板）"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset) PRESET="$2"; shift 2 ;;
        --help) show_help; exit 0 ;;
        *)
            echo "错误: 未知参数: $1"
            echo "提示: 构建请用 scripts/build.sh，测试请用 scripts/test.sh"
            show_help
            exit 1
            ;;
    esac
done

BUILD_DIR="${PROJECT_ROOT}/build/${PRESET}"

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "错误: 构建目录未配置: ${BUILD_DIR}" >&2
    echo "请先执行: bash scripts/build.sh --preset ${PRESET}" >&2
    exit 1
fi

echo "=== cmake --install ==="
echo "preset:    ${PRESET}"
echo "build dir: ${BUILD_DIR}"

cmake --install "${BUILD_DIR}"

INSTALL_DIR="$(grep -E '^CMAKE_INSTALL_PREFIX:' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2)"
INSTALL_TESTS_FLAG="$(grep -E '^INSTALL_TESTS:' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2)"
INSTALL_BENCHMARKS_FLAG="$(grep -E '^INSTALL_BENCHMARKS:' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2)"
echo ""
echo "安装完成: ${INSTALL_DIR}"
echo "  bin/                      demo 可执行文件"
echo "  lib/                      库（libdeploy_percept_core.a、libdeploy_percept_utils.a、VIPLite 等）"
echo "  include/deploy_percept/"
echo "  share/percept/apps/       示例与测试 fixture 数据"
if [[ "${INSTALL_TESTS_FLAG}" == "ON" ]]; then
    echo "  share/percept/tests/      测试可执行文件"
else
    echo "  (INSTALL_TESTS=OFF，未安装 share/percept/tests/)"
fi
if [[ "${INSTALL_BENCHMARKS_FLAG}" == "ON" ]]; then
    echo "  share/percept/benchmarks/ benchmark 可执行文件"
else
    echo "  (INSTALL_BENCHMARKS=OFF 或 ENABLE_BENCHMARKS=OFF，未安装 benchmarks/)"
fi
echo ""
echo "验证 install 包: bash scripts/test.sh --install-dir ${INSTALL_DIR}"
if [[ "${INSTALL_BENCHMARKS_FLAG}" == "ON" ]]; then
    echo "含性能测试:      bash scripts/test.sh --install-dir ${INSTALL_DIR} --bench 50"
fi
