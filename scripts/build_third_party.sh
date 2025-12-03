#!/bin/bash
# bash scripts/build_third_party.sh x86_64 --libs gtest,opencv
# bash scripts/build_third_party.sh aarch64 --libs gtest,opencv
# 为不同平台编译第三方库的通用脚本

set -e  # 遇到错误时停止执行

# 初始化变量
LIBS_TO_BUILD="all"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --libs)
            LIBS_TO_BUILD="$2"
            shift 2
            ;;
        -*)
            echo "未知选项: $1"
            echo "用法: $0 <platform> [--libs <libraries>]"
            echo "  platform: 目标平台 (aarch64, x86_64)"
            echo "  libraries: 逗号分隔的库列表 (例如: gtest,opencv)"
            echo "             默认构建所有支持的库"
            exit 1
            ;;
        *)
            if [ -z "$PLATFORM" ]; then
                PLATFORM="$1"
            else
                echo "未知参数: $1"
                echo "用法: $0 <platform> [--libs <libraries>]"
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查平台参数
if [ -z "$PLATFORM" ]; then
    echo "用法: $0 <platform> [--libs <libraries>]"
    echo "支持的平台: aarch64, x86_64"
    echo "支持的库: gtest, opencv"
    exit 1
fi

# 获取项目根目录
PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
echo "项目根目录: $PROJECT_ROOT"

# 根据平台设置相关变量
case "${PLATFORM}" in
    aarch64)
        # 设置交叉编译工具链
        CROSS_COMPILE_PREFIX=aarch64-linux-gnu
        TOOLCHAIN_FILE=${PROJECT_ROOT}/cmake/aarch64-toolchain.cmake
        ;;
    x86_64)
        CROSS_COMPILE_PREFIX=x86_64-linux-gnu
        TOOLCHAIN_FILE=${PROJECT_ROOT}/cmake/x86_64-toolchain.cmake
        ;;
    *)
        echo "错误: 不支持的平台 '${PLATFORM}'"
        echo "支持的平台: aarch64, x86_64"
        exit 1
        ;;
esac

# 创建平台对应的第三方库目录
mkdir -p ${PROJECT_ROOT}/tmp
INSTALL_DIR=${PROJECT_ROOT}/third_party/

echo "开始为${PLATFORM}平台编译第三方库..."
echo "安装目录: $INSTALL_DIR"
echo "使用交叉编译工具链: $CROSS_COMPILE_PREFIX"
echo "使用工具链文件: $TOOLCHAIN_FILE"
echo "构建的库: $LIBS_TO_BUILD"

# 设置编译器变量
export CC=${CROSS_COMPILE_PREFIX}-gcc
export CXX=${CROSS_COMPILE_PREFIX}-g++

# 检查交叉编译工具是否安装
if ! command -v ${CXX} &> /dev/null
then
    echo "错误: 未找到交叉编译工具链 $CXX"
    case "${PLATFORM}" in
        aarch64)
            echo "请先安装: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
            ;;
        x86_64)
            echo "请先安装: sudo apt install gcc-x86-64-linux-gnu g++-x86-64-linux-gnu"
            ;;
    esac
    exit 1
fi

echo "交叉编译工具链检查通过"

# 解析要构建的库列表
if [ "$LIBS_TO_BUILD" = "all" ]; then
    BUILD_GTEST=yes
    BUILD_OPENCV=yes
else
    BUILD_GTEST=no
    BUILD_OPENCV=no
    
    IFS=',' read -ra LIBS <<< "$LIBS_TO_BUILD"
    for lib in "${LIBS[@]}"; do
        case "$lib" in
            gtest)
                BUILD_GTEST=yes
                ;;
            opencv)
                BUILD_OPENCV=yes
                ;;
            *)
                echo "警告: 忽略未知的库 '$lib'"
                ;;
        esac
    done
fi

# 编译GTest（如果需要）
if [ "$BUILD_GTEST" = "yes" ]; then
    echo "开始编译GTest..."
    cd ${PROJECT_ROOT}/tmp
    # git clone https://gitee.com/mirrors/googletest.git -b v1.14.0
    cd ${PROJECT_ROOT}/tmp/googletest
    rm -rf build_${PLATFORM}
    mkdir -p build_${PLATFORM} && cd build_${PLATFORM}

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/gtest/${PLATFORM} \
        -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
        
    make -j$(nproc)
    make install

    echo "GTest编译完成"
else
    echo "跳过GTest编译"
fi

# 编译OpenCV（如果需要）
if [ "$BUILD_OPENCV" = "yes" ]; then
    echo "开始编译OpenCV..."
    cd ${PROJECT_ROOT}/tmp
    # git clone https://gitee.com/opencv/opencv.git
    cd ${PROJECT_ROOT}/tmp/opencv
    # git checkout 4.10.0
    rm -rf build_${PLATFORM}
    mkdir -p build_${PLATFORM} && cd build_${PLATFORM}

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/opencv/${PLATFORM} \
        -DOPENCV_DOWNLOAD_PATH=../../opencv_${PLATFORM}_cache \
        -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
        -DBUILD_SHARED_LIBS=OFF

    make -j4  

    make install

    echo "OpenCV编译完成"
else
    echo "跳过OpenCV编译"
fi

echo "为${PLATFORM}平台的第三方库构建任务已完成"
echo "已安装到 ${INSTALL_DIR}"