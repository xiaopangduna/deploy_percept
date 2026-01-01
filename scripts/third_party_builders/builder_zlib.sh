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
    echo "  --project-root <路径>   项目根目录 (默认: 当前目录)"
    echo "  --install-dir <路径>    安装目录 (默认: \$PROJECT_ROOT/third_party)"
    echo "  --help                 显示帮助"
}

PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"; shift 2 ;;
        --project-root)
            PROJECT_ROOT="$2"; shift 2 ;;
        --install-dir)
            INSTALL_DIR="$2"; shift 2 ;;
        --help)
            show_help; exit 0 ;;
        *)
            echo "错误: 未知参数 $1"; exit 1 ;;
    esac
done

if [ -z "$PLATFORM" ]; then
    echo "错误: 必须指定 --platform"
    exit 1
fi

case "$PLATFORM" in
    aarch64) CROSS_COMPILE_PREFIX="aarch64-linux-gnu" ;;
    x86_64)  CROSS_COMPILE_PREFIX="x86_64-linux-gnu" ;;
    *)
        echo "错误: 不支持的平台 $PLATFORM"
        exit 1 ;;
esac

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
INSTALL_DIR=${INSTALL_DIR:-${PROJECT_ROOT}/third_party}

export CC=${CROSS_COMPILE_PREFIX}-gcc
export AR=${CROSS_COMPILE_PREFIX}-ar
export RANLIB=${CROSS_COMPILE_PREFIX}-ranlib

echo "[zlib构建器] 平台: $PLATFORM"
echo "[zlib构建器] 安装路径: ${INSTALL_DIR}/zlib/${PLATFORM}"

mkdir -p ${PROJECT_ROOT}/tmp
cd ${PROJECT_ROOT}/tmp

if [ ! -d zlib ]; then
    echo "[zlib构建器] 克隆 zlib..."
    git clone https://github.com/madler/zlib.git
fi

cd zlib
rm -rf build_${PLATFORM}
mkdir build_${PLATFORM}
cd build_${PLATFORM}

../configure \
    --prefix=${INSTALL_DIR}/zlib/${PLATFORM} \
    --static

make -j$(nproc)
make install

echo "[zlib构建器] ✓ zlib 构建完成"
