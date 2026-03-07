#!/bin/bash
# 第三方库构建器：zlib
# 可独立运行，也可被 builder_cnpy.sh / third_party_builder.sh 调用
# 支持 host / cross 两种模式

set -e

show_help() {
    echo "zlib 构建器脚本"
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
    echo "  --toolchain-file <文件>     仅为接口统一而接受（zlib 不使用 CMake）"
    echo "  --help                      显示帮助"
    echo ""
    echo "说明:"
    echo "  - zlib 使用 configure/make，不使用 CMake"
    echo "  - cross 模式通过 CC / AR / RANLIB 实现"
    echo ""
    echo "示例:"
    echo "  # host 模式（本机架构）"
    echo "  bash scripts/third_party_builders/builder_zlib.sh --build-mode host"
    echo ""
    echo "  # cross 模式，编译 aarch64"
    echo "  bash scripts/third_party_builders/builder_zlib.sh --build-mode cross --platform aarch64"
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
# 项目根目录
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
    echo "[zlib构建器] Host 模式，检测到本机架构: $TARGET_ARCH"

    # 使用系统默认工具链
    CC=$(command -v gcc)
    AR=$(command -v ar)
    RANLIB=$(command -v ranlib)

elif [ "$BUILD_MODE" = "cross" ]; then
    if [ -z "$PLATFORM" ]; then
        echo "错误: cross 模式必须指定 --platform"
        show_help
        return 1
    fi

    TARGET_ARCH="$PLATFORM"

    case "$PLATFORM" in
        aarch64)
            CROSS_COMPILE_PREFIX="aarch64-linux-gnu"
            ;;
        x86_64)
            CROSS_COMPILE_PREFIX="x86_64-linux-gnu"
            ;;
        *)
            echo "错误: 不支持的平台 $PLATFORM"
            return 1
            ;;
    esac

    export CC=${CROSS_COMPILE_PREFIX}-gcc
    export AR=${CROSS_COMPILE_PREFIX}-ar
    export RANLIB=${CROSS_COMPILE_PREFIX}-ranlib

    echo "[zlib构建器] Cross 模式"
    echo "[zlib构建器] 目标架构: $TARGET_ARCH"
    echo "[zlib构建器] CC=$CC"

    if ! command -v $CC &>/dev/null; then
        echo "错误: 找不到交叉编译器 $CC"
        return 1
    fi
else
    echo "错误: 无效的 build-mode: $BUILD_MODE"
    show_help
    return 1
fi

# toolchain-file 仅提示
if [ -n "$TOOLCHAIN_FILE" ]; then
    echo "[zlib构建器] 提示: zlib 不使用 CMake，忽略 --toolchain-file=$TOOLCHAIN_FILE"
fi

echo "[zlib构建器] 项目根目录: $PROJECT_ROOT"
echo "[zlib构建器] 安装路径: ${INSTALL_DIR}/zlib/${TARGET_ARCH}"

# -----------------------------
# 准备源码
# -----------------------------
mkdir -p "${PROJECT_ROOT}/tmp"
cd "${PROJECT_ROOT}/tmp"

if [ ! -d zlib ]; then
    echo "[zlib构建器] 克隆 zlib..."
    git clone https://github.com/madler/zlib.git
else
    echo "[zlib构建器] zlib 已存在，跳过克隆"
fi

cd zlib

# -----------------------------
# 构建
# -----------------------------
rm -rf build_${TARGET_ARCH}
mkdir build_${TARGET_ARCH}
cd build_${TARGET_ARCH}

echo "[zlib构建器] 配置 zlib..."
../configure \
    --prefix="${INSTALL_DIR}/zlib/${TARGET_ARCH}" \
    --static

echo "[zlib构建器] 编译 zlib..."
make -j$(nproc)

echo "[zlib构建器] 安装 zlib..."
make install

# -----------------------------
# 验证
# -----------------------------
if [ -f "${INSTALL_DIR}/zlib/${TARGET_ARCH}/lib/libz.a" ]; then
    echo "[zlib构建器] ✓ zlib 构建完成"
    echo "[zlib构建器] lib: ${INSTALL_DIR}/zlib/${TARGET_ARCH}/lib/libz.a"
    echo "[zlib构建器] include: ${INSTALL_DIR}/zlib/${TARGET_ARCH}/include"
else
    echo "[zlib构建器] ✗ 构建失败，未找到 libz.a"
    return 1
fi
