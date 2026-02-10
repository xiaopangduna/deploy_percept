#!/bin/bash
# 第三方库构建器：OpenCV
# 可以单独运行，也可以由 third_party_builder.sh 调用
# 支持 host 模式（本机编译）和 cross 模式（交叉编译）

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 显示帮助信息
show_help() {
    echo "OpenCV 构建器脚本"
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
    echo "  --install-dir <路径>        安装目录 (默认: \$PROJECT_ROOT/third_party)"
    echo "  --toolchain-file <文件>     CMake工具链文件 (仅 cross 模式有效)"
    echo "  --help                      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # host 模式编译本机架构"
    echo "  bash scripts/third_party_builders/builder_opencv.sh --build-mode host"
    echo ""
    echo "  # cross 模式编译 aarch64"
    echo "  bash scripts/third_party_builders/builder_opencv.sh \\"
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

# 设置项目根目录默认值
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
echo "[OpenCV构建器] 项目根目录: $PROJECT_ROOT"
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误: 项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

# 设置安装目录默认值
INSTALL_DIR=${INSTALL_DIR:-${PROJECT_ROOT}/third_party}

# 根据构建模式设置平台和交叉编译
if [ "$BUILD_MODE" = "host" ]; then
    TARGET_ARCH=$(uname -m)
    echo "[OpenCV构建器] Host 模式，检测本机架构: $TARGET_ARCH"
    CROSS_COMPILE_PREFIX=""
elif [ "$BUILD_MODE" = "cross" ]; then
    if [ -z "$PLATFORM" ]; then
        echo "错误: cross 模式必须指定 --platform"
        show_help
        exit 1
    fi
    TARGET_ARCH="$PLATFORM"
    case "$PLATFORM" in
        aarch64) CROSS_COMPILE_PREFIX="aarch64-linux-gnu" ;;
        x86_64) CROSS_COMPILE_PREFIX="x86_64-linux-gnu" ;;
        *)
            echo "错误: 不支持的平台 '$PLATFORM'"
            exit 1
            ;;
    esac
    echo "[OpenCV构建器] Cross 模式，目标架构: $TARGET_ARCH"
else
    echo "错误: 无效的 build-mode: $BUILD_MODE"
    show_help
    exit 1
fi

# 设置工具链文件（仅 cross 模式有效）
if [ "$BUILD_MODE" = "cross" ]; then
    if [ -z "$TOOLCHAIN_FILE" ]; then
        TOOLCHAIN_FILE="${PROJECT_ROOT}/cmake/${PLATFORM}-toolchain.cmake"
        echo "[OpenCV构建器] 使用默认工具链文件: $TOOLCHAIN_FILE"
    fi
    if [ ! -f "$TOOLCHAIN_FILE" ]; then
        echo "警告: 工具链文件不存在: $TOOLCHAIN_FILE"
        echo "       CMake配置可能失败或使用系统默认编译器"
    fi
fi

# 设置编译器变量
if [ "$BUILD_MODE" = "cross" ]; then
    export CC=${CROSS_COMPILE_PREFIX}-gcc
    export CXX=${CROSS_COMPILE_PREFIX}-g++
    if ! command -v ${CXX} &> /dev/null; then
        echo "警告: 未找到交叉编译工具链 $CXX"
        echo "       CMake配置可能失败"
    fi
else
    CC=$(command -v gcc)
    CXX=$(command -v g++)
fi

echo "[OpenCV构建器] 使用编译器: CC=$CC, CXX=$CXX"
echo "[OpenCV构建器] 安装路径: $INSTALL_DIR/opencv/$TARGET_ARCH"

# 下载和构建OpenCV
mkdir -p "${PROJECT_ROOT}/tmp"
mkdir -p "${INSTALL_DIR}"

cd "${PROJECT_ROOT}/tmp"

# 克隆或更新代码
if [ ! -d "opencv" ]; then
    echo "[OpenCV构建器] 克隆OpenCV代码..."
    git clone https://gitee.com/opencv/opencv.git
else
    echo "[OpenCV构建器] OpenCV目录已存在，跳过克隆"
fi

cd opencv

# 切换到指定版本
echo "[OpenCV构建器] 切换到版本 4.5.4..."
git checkout 4.5.4

# 清理旧的构建目录
rm -rf build_${TARGET_ARCH}
mkdir -p build_${TARGET_ARCH}
cd build_${TARGET_ARCH}

# 创建下载缓存目录
OPENCV_CACHE_DIR="../../opencv_${TARGET_ARCH}_cache"
mkdir -p ${OPENCV_CACHE_DIR}

# 配置CMake
echo "[OpenCV构建器] 配置CMake..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/opencv/${TARGET_ARCH} \
-DOPENCV_DOWNLOAD_PATH=${OPENCV_CACHE_DIR} \
-DBUILD_SHARED_LIBS=OFF \
-DBUILD_PNG=ON \
-DWITH_EIGEN=OFF \
-DPNG_LIBRARY= \
-DPNG_PNG_INCLUDE_DIR="

if [ "$BUILD_MODE" = "cross" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}"
fi

cmake .. $CMAKE_ARGS

# 编译和安装
# CPU_CORES=$(nproc 2>/dev/null || echo 4)
CPU_CORES=4
echo "[OpenCV构建器] 使用 $CPU_CORES 个CPU核心进行编译"
make -j$CPU_CORES
make install

# 验证安装
if [ -f "${INSTALL_DIR}/opencv/${TARGET_ARCH}/lib/libopencv_core.a" ] || \
   [ -f "${INSTALL_DIR}/opencv/${TARGET_ARCH}/lib64/libopencv_core.a" ]; then
    echo "[OpenCV构建器] ✓ OpenCV静态库安装成功"
elif [ -f "${INSTALL_DIR}/opencv/${TARGET_ARCH}/lib/libopencv_core.so" ] || \
     [ -f "${INSTALL_DIR}/opencv/${TARGET_ARCH}/lib64/libopencv_core.so" ]; then
    echo "[OpenCV构建器] ✓ OpenCV动态库安装成功"
else
    echo "[OpenCV构建器] ⚠ 警告: 找不到OpenCV核心库文件，但安装命令已成功执行"
fi

echo "[OpenCV构建器] OpenCV构建完成"
