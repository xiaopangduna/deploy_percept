#!/bin/bash
# 第三方库构建器：spdlog
# 可以单独运行，也可以由 third_party_builder.sh 调用
# 支持 host 模式（本机编译）和 cross 模式（交叉编译）

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 显示帮助信息
show_help() {
    echo "spdlog 构建器脚本"
    echo ""
    echo "用法:"
    echo "  $0 [选项]"
    echo ""
    echo "必需选项:"
    echo "  --build-mode <host|cross>  构建模式：host（本机编译）或 cross（交叉编译） [必需]"
    echo ""
    echo "可选选项:"
    echo "  --platform <平台>           目标平台，仅 cross 模式下有效 (aarch64, x86_64)"
    echo "  --project-root <路径>       项目根目录 (默认: 当前目录)"
    echo "  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)"
    echo "  --toolchain-file <文件>    CMake工具链文件 (默认: \$PROJECT_ROOT/cmake/\$PLATFORM-toolchain.cmake, cross 模式有效)"
    echo "  --help                     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # host 模式"
    echo "  bash scripts/third_party_builders/builder_spdlog.sh --build-mode host"
    echo ""
    echo "  # cross 模式 aarch64"
    echo "  bash scripts/third_party_builders/builder_spdlog.sh \\"
    echo "    --build-mode cross \\"
    echo "    --platform aarch64 \\"
    echo "    --project-root /path/to/project"
}

# 初始化变量
BUILD_MODE=""
PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""
TOOLCHAIN_FILE=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-mode)
            BUILD_MODE="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --project-root)
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --toolchain-file)
            TOOLCHAIN_FILE="$2"
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

# 检查必需参数
if [ -z "$BUILD_MODE" ]; then
    echo "错误: 必须指定构建模式 (--build-mode)"
    show_help
    exit 1
fi

# 项目根目录
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
echo "[spdlog构建器] 项目根目录: $PROJECT_ROOT"
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误: 项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

# 安装目录
INSTALL_DIR=${INSTALL_DIR:-${PROJECT_ROOT}/third_party}

# 构建模式
if [ "$BUILD_MODE" = "host" ]; then
    TARGET_ARCH=$(uname -m)
    CROSS_COMPILE_PREFIX=""
    echo "[spdlog构建器] Host 模式，架构: $TARGET_ARCH"
elif [ "$BUILD_MODE" = "cross" ]; then
    if [ -z "$PLATFORM" ]; then
        echo "错误: cross 模式必须指定 --platform"
        exit 1
    fi
    TARGET_ARCH="$PLATFORM"
    case "$PLATFORM" in
        aarch64) CROSS_COMPILE_PREFIX="aarch64-linux-gnu" ;;
        x86_64)  CROSS_COMPILE_PREFIX="x86_64-linux-gnu" ;;
        *)
            echo "错误: 不支持的平台: $PLATFORM"
            exit 1
            ;;
    esac
    echo "[spdlog构建器] Cross 模式，目标架构: $TARGET_ARCH"
else
    echo "错误: 无效的 build-mode: $BUILD_MODE"
    exit 1
fi

# 工具链
if [ "$BUILD_MODE" = "cross" ]; then
    if [ -z "$TOOLCHAIN_FILE" ]; then
        TOOLCHAIN_FILE="${PROJECT_ROOT}/cmake/${PLATFORM}-toolchain.cmake"
        echo "[spdlog构建器] 使用默认工具链文件: $TOOLCHAIN_FILE"
    fi
    if [ ! -f "$TOOLCHAIN_FILE" ]; then
        echo "警告: 工具链文件不存在: $TOOLCHAIN_FILE"
    fi
fi

# 编译器
if [ "$BUILD_MODE" = "cross" ]; then
    export CC=${CROSS_COMPILE_PREFIX}-gcc
    export CXX=${CROSS_COMPILE_PREFIX}-g++
else
    CC=$(command -v gcc)
    CXX=$(command -v g++)
fi

echo "[spdlog构建器] 编译器: CC=$CC, CXX=$CXX"
echo "[spdlog构建器] 安装路径: $INSTALL_DIR/spdlog/$TARGET_ARCH"

# 下载 & 构建
mkdir -p "${PROJECT_ROOT}/tmp"
cd "${PROJECT_ROOT}/tmp"

# 获取源码
if [ ! -d "spdlog" ]; then
    echo "[spdlog构建器] 克隆 spdlog..."
    git clone https://github.com/gabime/spdlog.git -b v1.13.0
else
    echo "[spdlog构建器] spdlog 已存在，跳过克隆"
fi

cd spdlog

# 构建目录
rm -rf build_${TARGET_ARCH}
mkdir build_${TARGET_ARCH}
cd build_${TARGET_ARCH}

# CMake 配置
CMAKE_ARGS="
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/spdlog/${TARGET_ARCH}
-DSPDLOG_BUILD_EXAMPLES=OFF
-DSPDLOG_BUILD_TESTS=OFF
-DSPDLOG_BUILD_BENCH=OFF
-DSPDLOG_FMT_EXTERNAL=OFF
"

if [ "$BUILD_MODE" = "cross" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}"
fi

echo "[spdlog构建器] 配置 CMake..."
cmake .. $CMAKE_ARGS

make -j$(nproc)
make install

# 校验
if [ -f "${INSTALL_DIR}/spdlog/${TARGET_ARCH}/lib/libspdlog.a" ] || \
   [ -f "${INSTALL_DIR}/spdlog/${TARGET_ARCH}/lib64/libspdlog.a" ]; then
    echo "[spdlog构建器] ✓ spdlog 安装成功"
else
    echo "[spdlog构建器] ⚠ 未找到 libspdlog.a，请检查安装结果"
fi

echo "[spdlog构建器] spdlog 构建完成"
