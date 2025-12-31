#!/bin/bash
# 第三方库构建器：GTest
# 可以单独运行，也可以由 build_third_party.sh 调用

set -e

# 显示帮助信息
show_help() {
    echo "GTest 构建器脚本"
    echo ""
    echo "用法:"
    echo "  $0 [选项]"
    echo ""
    echo "必需选项:"
    echo "  --platform <平台>          目标平台 (aarch64, x86_64) [必需]"
    echo ""
    echo "可选选项:"
    echo "  --project-root <路径>      项目根目录 (默认: 当前目录)"
    echo "  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)"
    echo "  --toolchain-file <文件>    CMake工具链文件 (默认: \$PROJECT_ROOT/cmake/\$PLATFORM-toolchain.cmake)"
    echo "  --help                    显示此帮助信息"
    echo ""
    echo ""
    echo "示例:"
    echo "  # 在项目根目录下运行"
    echo "  cd /path/to/project"
    echo "  bash scripts/third_party_builders/builder_gtest.sh --platform x86_64"
    echo ""
    echo "  # 从任何位置运行"
    echo "  bash /path/to/project/scripts/third_party_builders/builder_gtest.sh \\"
    echo "    --platform aarch64 \\"
    echo "    --project-root /path/to/project"
}

# 初始化变量
PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""
TOOLCHAIN_FILE=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --toolchain-file)
            TOOLCHAIN_FILE="$2"
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
if [ -z "$PLATFORM" ]; then
    echo "错误: 必须指定平台 (--platform)"
    show_help
    exit 1
fi

# 验证平台参数并设置交叉编译前缀
case "${PLATFORM}" in
    aarch64)
        CROSS_COMPILE_PREFIX="aarch64-linux-gnu"
        ;;
    x86_64)
        CROSS_COMPILE_PREFIX="x86_64-linux-gnu"
        ;;
    *)
        echo "错误: 不支持的平台 '$PLATFORM'"
        echo "支持的平台: aarch64, x86_64"
        exit 1
        ;;
esac

echo "[GTest构建器] 平台: $PLATFORM"
echo "[GTest构建器] 交叉编译前缀: $CROSS_COMPILE_PREFIX"

# 设置项目根目录默认值（当前工作目录）
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
echo "[GTest构建器] 项目根目录: $PROJECT_ROOT"

# 检查项目根目录是否合理
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误: 项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

# 设置安装目录默认值
INSTALL_DIR=${INSTALL_DIR:-${PROJECT_ROOT}/third_party}

# 设置工具链文件默认值
if [ -z "$TOOLCHAIN_FILE" ]; then
    TOOLCHAIN_FILE="${PROJECT_ROOT}/cmake/${PLATFORM}-toolchain.cmake"
    echo "[GTest构建器] 使用默认工具链文件: $TOOLCHAIN_FILE"
fi

# 检查工具链文件是否存在
if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo "警告: 工具链文件不存在: $TOOLCHAIN_FILE"
    echo "       CMake配置可能失败或使用系统默认编译器"
fi

# 设置编译器变量
export CC=${CROSS_COMPILE_PREFIX}-gcc
export CXX=${CROSS_COMPILE_PREFIX}-g++

# 检查编译器是否存在（仅警告，不终止）
if ! command -v ${CXX} &> /dev/null; then
    echo "警告: 未找到交叉编译工具链 $CXX"
    echo "       CMake配置可能失败"
    echo "       x86_64平台请安装: sudo apt install gcc-x86-64-linux-gnu g++-x86-64-linux-gnu"
    echo "       aarch64平台请安装: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
fi

echo "[GTest构建器] 开始构建GTest库"
echo "  平台: $PLATFORM"
echo "  安装路径: $INSTALL_DIR/gtest/$PLATFORM"
echo "  工具链文件: $TOOLCHAIN_FILE"

# 下载和构建GTest
mkdir -p ${PROJECT_ROOT}/tmp
mkdir -p ${PROJECT_ROOT}/third_party

cd ${PROJECT_ROOT}/tmp

# 克隆或更新代码
if [ ! -d "googletest" ]; then
    echo "[GTest构建器] 克隆GTest代码..."
    git clone https://gitee.com/mirrors/googletest.git -b v1.14.0
    if [ $? -ne 0 ]; then
        echo "[GTest构建器] 错误: 克隆GTest代码失败"
        exit 1
    fi
else
    echo "[GTest构建器] GTest目录已存在，跳过克隆"
fi

cd googletest

# 清理旧的构建目录
rm -rf build_${PLATFORM}
mkdir -p build_${PLATFORM}
cd build_${PLATFORM}

# 配置和构建
echo "[GTest构建器] 配置CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/gtest/${PLATFORM} \
    -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}

if [ $? -ne 0 ]; then
    echo "[GTest构建器] 错误: GTest配置失败"
    exit 1
fi

echo "[GTest构建器] 编译GTest..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "[GTest构建器] 错误: GTest编译失败"
    exit 1
fi

echo "[GTest构建器] 安装GTest..."
make install

if [ $? -ne 0 ]; then
    echo "[GTest构建器] 错误: GTest安装失败"
    exit 1
fi

# 验证安装
if [ -f "${INSTALL_DIR}/gtest/${PLATFORM}/lib/libgtest.a" ] || \
   [ -f "${INSTALL_DIR}/gtest/${PLATFORM}/lib64/libgtest.a" ]; then
    echo "[GTest构建器] ✓ GTest安装成功"
else
    echo "[GTest构建器] ⚠ 警告: 找不到GTest库文件，但安装命令已成功执行"
    echo "[GTest构建器]   提示: 某些系统可能将库安装到不同位置"
fi

echo "[GTest构建器] GTest构建完成"