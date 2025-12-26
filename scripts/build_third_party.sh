#!/bin/bash
# bash scripts/build_third_party.sh x86_64 --libs gtest,opencv,cnpy
# bash scripts/build_third_party.sh aarch64 --libs gtest,opencv,cnpy
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
            echo "  libraries: 逗号分隔的库列表 (例如: gtest,opencv,rknpu,spdlog,cnpy"
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
    echo "支持的库: gtest, opencv, spdlog, rknpu, cnpy"
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

# 检查zlib开发库是否安装
if [ "${PLATFORM}" = "x86_64" ]; then
    if ! pkg-config --exists zlib; then
        echo "错误: 未找到zlib开发库"
        echo "请先安装: sudo apt install zlib1g-dev"
        exit 1
    fi
else
    # 对于aarch64交叉编译，检查aarch64版本的zlib
    if [ ! -f "/usr/lib/$(uname -m)-linux-gnu/aarch64-linux-gnu/libz.a" ] && [ ! -f "/usr/aarch64-linux-gnu/lib/libz.a" ]; then
        echo "警告: 可能缺少aarch64的zlib库，如果编译失败请确认已安装aarch64的zlib开发包"
    fi
fi

# 解析要构建的库列表
if [ "$LIBS_TO_BUILD" = "all" ]; then
    BUILD_GTEST=yes
    BUILD_OPENCV=yes
    BUILD_SPDLOG=yes
    BUILD_RKNPU=yes
    BUILD_CNPY=yes

else
    BUILD_GTEST=no
    BUILD_OPENCV=no
    BUILD_RKNPU=no
    BUILD_SPDLOG=no
    BUILD_CNPY=no

    IFS=',' read -ra LIBS <<< "$LIBS_TO_BUILD"
    for lib in "${LIBS[@]}"; do
        case "$lib" in
            gtest)
                BUILD_GTEST=yes
                ;;
            opencv)
                BUILD_OPENCV=yes
                ;;
            spdlog)
                BUILD_SPDLOG=yes
                ;;
            rknpu)
                BUILD_RKNPU=yes
                ;;
            cnpy)
                BUILD_CNPY=yes
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
    if [ ! -d "googletest" ]; then
        git clone https://gitee.com/mirrors/googletest.git -b v1.14.0
    else
        echo "googletest目录已存在，跳过克隆"
    fi
    cd ${PROJECT_ROOT}/tmp/googletest
    rm -rf build_${PLATFORM}
    mkdir -p build_${PLATFORM} && cd build_${PLATFORM}
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/gtest/${PLATFORM} \
        -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}
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
    if [ ! -d "opencv" ]; then
        git clone https://gitee.com/opencv/opencv.git
    else
        echo "opencv目录已存在，跳过克隆"
    fi
    cd ${PROJECT_ROOT}/tmp/opencv
    git checkout 4.5.4
    rm -rf build_${PLATFORM}
    mkdir -p build_${PLATFORM} && cd build_${PLATFORM}
    # 当系统libpng异常才开启opencv自带的libpng
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/opencv/${PLATFORM} \
        -DOPENCV_DOWNLOAD_PATH=../../opencv_${PLATFORM}_cache \
        -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_PNG=ON \
        -DPNG_LIBRARY="" \
        -DPNG_PNG_INCLUDE_DIR=""
    make -j4  
    make install

    echo "OpenCV编译完成"
else
    echo "跳过OpenCV编译"
fi

# 编译spdlog（如果需要）
if [ "$BUILD_SPDLOG" = "yes" ]; then
    echo "开始编译spdlog..."
    cd ${PROJECT_ROOT}/tmp
    if [ ! -d "spdlog" ]; then
        git clone https://gitee.com/mirror-luyi/spdlog.git
        cd ${PROJECT_ROOT}/tmp/spdlog
        git checkout v1.14.1
    else
        echo "spdlog目录已存在，跳过git clone步骤"
    fi
    cd ${PROJECT_ROOT}/tmp/spdlog
    rm -rf build_${PLATFORM}
    mkdir -p build_${PLATFORM} && cd build_${PLATFORM}

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/spdlog/${PLATFORM} \
        -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} 

    make -j4
    make install

    echo "spdlog编译完成"
else
    echo "跳过spdlog编译"
fi

# 编译cnpy（如果需要）
if [ "$BUILD_CNPY" = "yes" ]; then
    echo "开始编译cnpy..."
    cd ${PROJECT_ROOT}/tmp
    if [ ! -d "cnpy" ]; then
        echo "克隆cnpy仓库..."
        git clone https://github.com/rogersce/cnpy.git
    else
        echo "cnpy目录已存在，跳过克隆"
    fi

    cd ${PROJECT_ROOT}/tmp/cnpy
    rm -rf build_${PLATFORM}
    mkdir -p build_${PLATFORM} && cd build_${PLATFORM}

    echo "配置cmake..."
    # 配置cmake并编译
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/cnpy/${PLATFORM} \
        -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
        -DBUILD_SHARED_LIBS=OFF

    echo "开始编译..."
    make -j$(nproc)
    echo "编译完成，开始安装..."
    make install

    echo "cnpy编译完成，已安装到 ${INSTALL_DIR}/cnpy/${PLATFORM}"
else
    echo "跳过cnpy编译"
fi

# ======================
# 新增：处理 RKNPU1/RKNPU2（仅 aarch64）
# ======================

if [ "${PLATFORM}" = "aarch64" ] && [ "${BUILD_RKNPU}" = "yes" ]; then
    echo "开始编译RKNPU..."
    BUILD_RKNPU=yes
else
    echo "跳过RKNPU编译"
    BUILD_RKNPU=no
fi

if [ "$BUILD_RKNPU" = "yes" ]; then
    echo "开始处理RKNPU库..."
    cd ${PROJECT_ROOT}/tmp

    if [ ! -d "rknn_model_zoo" ]; then
        echo "初始化rknn_model_zoo仓库..."
        git init rknn_model_zoo
        cd rknn_model_zoo
        echo "添加远程仓库..."
        git remote add origin https://github.com/airockchip/rknn_model_zoo.git
        echo "初始化sparse-checkout..."
        git sparse-checkout init --cone
        echo "设置要检出的目录..."
        git sparse-checkout set 3rdparty/rknpu2 3rdparty/rknpu1
        echo "拉取main分支..."
        git pull origin main
        echo "仓库克隆完成"
    else
        echo "rknn_model_zoo目录已存在，跳过克隆"
    fi

    cd ${PROJECT_ROOT}/tmp/rknn_model_zoo
    echo "开始拷贝RKNPU库文件到third_party目录..."
    # 拷贝rknpu2和rknpu1目录到third_party
    if [ -d "3rdparty/rknpu2" ]; then
        echo "拷贝rknpu2目录..."
        # 如果目标目录已存在，则先删除再拷贝
        if [ -d "${INSTALL_DIR}/rknpu2" ]; then
            echo "删除已存在的${INSTALL_DIR}/rknpu2目录..."
            rm -rf ${INSTALL_DIR}/rknpu2
        fi
        cp -r 3rdparty/rknpu2 ${INSTALL_DIR}/
    fi
    
    if [ -d "3rdparty/rknpu1" ]; then
        echo "拷贝rknpu1目录..."
        # 如果目标目录已存在，则先删除再拷贝
        if [ -d "${INSTALL_DIR}/rknpu1" ]; then
            echo "删除已存在的${INSTALL_DIR}/rknpu1目录..."
            rm -rf ${INSTALL_DIR}/rknpu1
        fi
        cp -r 3rdparty/rknpu1 ${INSTALL_DIR}/
    fi
    
    echo "RKNPU库处理完成"
fi

echo "为${PLATFORM}平台的第三方库构建任务已完成"
echo "已安装到 ${INSTALL_DIR}"