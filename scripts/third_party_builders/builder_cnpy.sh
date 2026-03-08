#!/bin/bash
# v0.0.1-2026.03.07
# 第三方库构建器：cnpy
# 可以单独运行，也可以由 build_third_party.sh 调用
# 支持 host / cross 两种模式

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# -----------------------------
# 帮助
# -----------------------------
show_help() {
    echo "cnpy 构建器脚本"
    echo ""
    echo "用法:"
    echo "  $0 --build-mode <host|cross> [选项]"
    echo ""
    echo "必需选项:"
    echo "  --build-mode <host|cross>   构建模式：host（本机）或 cross（交叉编译）"
    echo ""
    echo "可选选项:"
    echo "  --platform <平台>           目标平台，仅 cross 模式有效 (aarch64, x86_64)"
    echo "  --project-root <路径>       项目根目录 (默认: 当前目录)"
    echo "  --install-dir <路径>        安装目录 (默认: \$PROJECT_ROOT/third_party)"
    echo "  --toolchain-file <文件>     CMake工具链文件（cross 模式有效）"
    echo "  --help                      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # host 模式（本机编译）"
    echo "  bash scripts/third_party_builders/builder_cnpy.sh --build-mode host"
    echo ""
    echo "  # cross 模式（aarch64）"
    echo "  bash scripts/third_party_builders/builder_cnpy.sh --build-mode cross --platform aarch64"
}

# -----------------------------
# 参数
# -----------------------------
BUILD_MODE=""
PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""
TOOLCHAIN_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-mode)   BUILD_MODE="$2"; shift 2 ;;
        --platform)     PLATFORM="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --install-dir)  INSTALL_DIR="$2"; shift 2 ;;
        --toolchain-file) TOOLCHAIN_FILE="$2"; shift 2 ;;
        --help) show_help; return 0 ;;
        *)
            echo "错误: 未知参数 $1"
            show_help
            return 1
            ;;
    esac
done

# -----------------------------
# 校验 build-mode
# -----------------------------
if [ -z "$BUILD_MODE" ]; then
    echo "错误: 必须指定 --build-mode"
    show_help
    return 1
fi

# -----------------------------
# 项目路径
# -----------------------------
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误: 项目根目录不存在: $PROJECT_ROOT"
    return 1
fi

INSTALL_DIR=${INSTALL_DIR:-${PROJECT_ROOT}/third_party}

# -----------------------------
# host / cross 分支
# -----------------------------
if [ "$BUILD_MODE" = "host" ]; then
    TARGET_ARCH=$(uname -m)
    echo "[cnpy构建器] Host 模式"
    echo "[cnpy构建器] 检测到本机架构: $TARGET_ARCH"

    CC=$(command -v gcc)
    CXX=$(command -v g++)

elif [ "$BUILD_MODE" = "cross" ]; then
    if [ -z "$PLATFORM" ]; then
        echo "错误: cross 模式必须指定 --platform"
        show_help
        return 1
    fi

    TARGET_ARCH="$PLATFORM"

    case "$PLATFORM" in
        aarch64) CROSS_COMPILE_PREFIX="aarch64-linux-gnu" ;;
        x86_64)  CROSS_COMPILE_PREFIX="x86_64-linux-gnu" ;;
        *)
            echo "错误: 不支持的平台 $PLATFORM"
            return 1
            ;;
    esac

    export CC=${CROSS_COMPILE_PREFIX}-gcc
    export CXX=${CROSS_COMPILE_PREFIX}-g++

    echo "[cnpy构建器] Cross 模式"
    echo "[cnpy构建器] 目标架构: $TARGET_ARCH"
    echo "[cnpy构建器] CC=$CC"

    if ! command -v $CXX &>/dev/null; then
        echo "错误: 找不到交叉编译工具链 $CXX"
        return 1
    fi
else
    echo "错误: 无效的 build-mode: $BUILD_MODE"
    show_help
    return 1
fi

# -----------------------------
# toolchain file（仅 cross）
# -----------------------------
if [ "$BUILD_MODE" = "cross" ]; then
    if [ -z "$TOOLCHAIN_FILE" ]; then
        TOOLCHAIN_FILE="${PROJECT_ROOT}/cmake/${PLATFORM}-toolchain.cmake"
        echo "[cnpy构建器] 使用默认工具链文件: $TOOLCHAIN_FILE"
    fi

    if [ ! -f "$TOOLCHAIN_FILE" ]; then
        echo "警告: 工具链文件不存在: $TOOLCHAIN_FILE"
    fi
fi

echo "[cnpy构建器] 安装路径: ${INSTALL_DIR}/cnpy/${TARGET_ARCH}"

# -----------------------------
# 构建 zlib 依赖
# -----------------------------
echo "[cnpy构建器] 构建依赖 zlib..."

if [ "$BUILD_MODE" = "host" ]; then
    bash "${SCRIPT_DIR}/builder_zlib.sh" \
        --build-mode host \
        --project-root "${PROJECT_ROOT}" \
        --install-dir "${INSTALL_DIR}"
else
    bash "${SCRIPT_DIR}/builder_zlib.sh" \
        --build-mode cross \
        --platform "${PLATFORM}" \
        --project-root "${PROJECT_ROOT}" \
        --install-dir "${INSTALL_DIR}"
fi

ZLIB_ROOT="${INSTALL_DIR}/zlib/${TARGET_ARCH}"

# -----------------------------
# 获取 cnpy 源码
# -----------------------------
mkdir -p "${PROJECT_ROOT}/tmp"
cd "${PROJECT_ROOT}/tmp"

if [ ! -d cnpy ]; then
    echo "[cnpy构建器] 克隆 cnpy..."
    git clone https://github.com/rogersce/cnpy.git
else
    echo "[cnpy构建器] cnpy 已存在，跳过克隆"
fi

cd cnpy

# -----------------------------
# 构建 cnpy
# -----------------------------
rm -rf build_${TARGET_ARCH}
mkdir build_${TARGET_ARCH}
cd build_${TARGET_ARCH}

echo "[cnpy构建器] 配置 CMake..."

CMAKE_ARGS="
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/cnpy/${TARGET_ARCH}
-DBUILD_SHARED_LIBS=OFF
-DZLIB_ROOT=${ZLIB_ROOT}
-DZLIB_LIBRARY=${ZLIB_ROOT}/lib/libz.a
-DZLIB_INCLUDE_DIR=${ZLIB_ROOT}/include
"

if [ "$BUILD_MODE" = "cross" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}"
fi

cmake .. ${CMAKE_ARGS}

echo "[cnpy构建器] 编译 cnpy..."
make -j$(nproc)

echo "[cnpy构建器] 安装 cnpy..."
make install

# -----------------------------
# 验证
# -----------------------------
if [ -f "${INSTALL_DIR}/cnpy/${TARGET_ARCH}/lib/libcnpy.a" ]; then
    echo "[cnpy构建器] ✓ cnpy 构建完成"
    echo "[cnpy构建器] lib: ${INSTALL_DIR}/cnpy/${TARGET_ARCH}/lib/libcnpy.a"
    echo "[cnpy构建器] include: ${INSTALL_DIR}/cnpy/${TARGET_ARCH}/include"
else
    echo "[cnpy构建器] ⚠ 未找到 libcnpy.a，请检查构建结果"
    return 1
fi

echo "[cnpy构建器] cnpy 构建完成"
