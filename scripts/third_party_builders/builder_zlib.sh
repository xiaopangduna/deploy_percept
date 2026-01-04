#!/bin/bash
# 第三方库构建器：zlib
# 可独立运行，也可被 builder_cnpy.sh 调用

set -e

show_help() {
    echo "zlib 构建器脚本"
    echo ""
    echo "用法:"
    echo "  $0 --platform <aarch64|x86_64> [选项]"
    echo ""
    echo "选项:"
    echo "  --platform <平台>          目标平台 (aarch64, x86_64 等)"
    echo "  --project-root <路径>     项目根目录 (默认: 当前目录)"
    echo "  --install-dir <路径>      安装目录 (默认: \$PROJECT_ROOT/third_party)"
    echo "  --toolchain-file <文件>   CMake工具链文件 (默认: \$PROJECT_ROOT/cmake/\$PLATFORM-toolchain.cmake)"
    echo "  --help                    显示帮助"
    echo ""
    echo "注意:"
    echo "  - zlib 使用传统的 configure/make 构建系统，不使用 CMake"
    echo "  - --toolchain-file 参数在此脚本中不被使用，仅为接口统一而接受"
    echo "  - 交叉编译通过环境变量 CC, AR, RANLIB 实现"
    echo ""
    echo "示例:"
    echo "  # 交叉编译 aarch64 版本"
    echo "  $0 --platform aarch64"
    echo ""
    echo "  # 指定项目根目录和安装目录"
    echo "  $0 --platform x86_64 --project-root /path/to/project --install-dir /path/to/install"
}

PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""
TOOLCHAIN_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "错误: 未知参数 $1"
            show_help
            exit 1
            ;;
    esac
done

if [ -z "$PLATFORM" ]; then
    echo "错误: 必须指定 --platform"
    show_help
    exit 1
fi

case "$PLATFORM" in
    aarch64) 
        CROSS_COMPILE_PREFIX="aarch64-linux-gnu"
        echo "[zlib构建器] 平台: aarch64 (ARM64)"
        ;;
    x86_64)  
        CROSS_COMPILE_PREFIX="x86_64-linux-gnu"
        echo "[zlib构建器] 平台: x86_64"
        ;;
    *)
        echo "错误: 不支持的平台 $PLATFORM"
        exit 1
        ;;
esac

# 设置项目根目录默认值
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}

# 检查项目根目录是否存在
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误: 项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

# 设置安装目录默认值
INSTALL_DIR=${INSTALL_DIR:-${PROJECT_ROOT}/third_party}

# 接受但不使用 toolchain-file 参数
if [ -n "$TOOLCHAIN_FILE" ]; then
    echo "[zlib构建器] 收到工具链文件参数: $TOOLCHAIN_FILE"
    echo "[zlib构建器] 注意: zlib 使用 configure/make 构建系统，不使用 CMake 工具链文件"
    echo "[zlib构建器] 交叉编译将通过环境变量 CC, AR, RANLIB 实现"
else
    # 如果未提供，使用默认路径（仅用于显示）
    DEFAULT_TOOLCHAIN_FILE="${PROJECT_ROOT}/cmake/${PLATFORM}-toolchain.cmake"
    echo "[zlib构建器] 未指定工具链文件，使用默认位置（如果存在）: $DEFAULT_TOOLCHAIN_FILE"
fi

echo "[zlib构建器] 项目根目录: $PROJECT_ROOT"
echo "[zlib构建器] 安装路径: ${INSTALL_DIR}/zlib/${PLATFORM}"

# 设置交叉编译环境变量
export CC=${CROSS_COMPILE_PREFIX}-gcc
export AR=${CROSS_COMPILE_PREFIX}-ar
export RANLIB=${CROSS_COMPILE_PREFIX}-ranlib

echo "[zlib构建器] 设置环境变量:"
echo "[zlib构建器]   CC=$CC"
echo "[zlib构建器]   AR=$AR"
echo "[zlib构建器]   RANLIB=$RANLIB"

# 检查交叉编译工具链是否可用
echo "[zlib构建器] 检查交叉编译工具链..."
if ! command -v $CC &> /dev/null; then
    echo "错误: 找不到交叉编译器 $CC"
    echo "请安装相应的交叉编译工具链，例如:"
    echo "  Ubuntu/Debian: sudo apt install gcc-${CROSS_COMPILE_PREFIX} g++-${CROSS_COMPILE_PREFIX}"
    exit 1
fi

# 创建临时目录
mkdir -p ${PROJECT_ROOT}/tmp
cd ${PROJECT_ROOT}/tmp

# 克隆或更新 zlib 仓库
if [ ! -d zlib ]; then
    echo "[zlib构建器] 克隆 zlib..."
    git clone https://github.com/madler/zlib.git
else
    echo "[zlib构建器] zlib 仓库已存在，跳过克隆"
    echo "[zlib构建器] 如需更新，请手动执行: cd ${PROJECT_ROOT}/tmp/zlib && git pull"
fi

cd zlib

# 清理之前的构建目录
if [ -d "build_${PLATFORM}" ]; then
    echo "[zlib构建器] 清理之前的构建目录: build_${PLATFORM}"
    rm -rf build_${PLATFORM}
fi

mkdir build_${PLATFORM}
cd build_${PLATFORM}

echo "[zlib构建器] 配置 zlib..."
echo "[zlib构建器] 安装前缀: ${INSTALL_DIR}/zlib/${PLATFORM}"
echo "[zlib构建器] 构建静态库: 是"

# 配置 zlib
../configure \
    --prefix=${INSTALL_DIR}/zlib/${PLATFORM} \
    --static

echo "[zlib构建器] 编译 zlib..."
make -j$(nproc)

echo "[zlib构建器] 安装 zlib..."
make install

# 验证安装
if [ -f "${INSTALL_DIR}/zlib/${PLATFORM}/lib/libz.a" ]; then
    echo "[zlib构建器] ✓ zlib 构建完成"
    echo "[zlib构建器] 库文件位置: ${INSTALL_DIR}/zlib/${PLATFORM}/lib/libz.a"
    echo "[zlib构建器] 头文件位置: ${INSTALL_DIR}/zlib/${PLATFORM}/include"
    
    # 显示构建的库信息
    echo "[zlib构建器] 库文件信息:"
    file ${INSTALL_DIR}/zlib/${PLATFORM}/lib/libz.a
    echo "[zlib构建器] 安装目录内容:"
    ls -la ${INSTALL_DIR}/zlib/${PLATFORM}/lib/
else
    echo "[zlib构建器] 错误: 安装后找不到库文件"
    exit 1
fi