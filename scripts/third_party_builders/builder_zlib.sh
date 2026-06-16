#!/bin/bash
# 第三方库构建器：zlib
# 可以单独运行，也可以由 build_third_party.sh 调用

set -e

# 显示帮助信息
show_help() {
    echo "zlib 构建器脚本"
    echo ""
    echo "用法:"
    echo "bash $0 [选项]"
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
    echo "注意:"
    echo "  - zlib 使用传统的 configure/make 构建系统，不使用 CMake"
    echo "  - --toolchain-file 参数在此脚本中不被使用，仅为接口统一而接受"
    echo "  - 交叉编译通过环境变量 CC, AR, RANLIB 实现"
    echo ""
    echo "示例:"
    echo "  # 在项目根目录下运行"
    echo "  cd /path/to/project"
    echo "  bash scripts/third_party_builders/builder_zlib.sh --platform x86_64"
    echo ""
    echo "  # 从任何位置运行"
    echo "  bash /path/to/project/scripts/third_party_builders/builder_zlib.sh \\"
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

echo "[zlib构建器] 平台: $PLATFORM"
echo "[zlib构建器] 交叉编译前缀: $CROSS_COMPILE_PREFIX"

# 设置项目根目录默认值（当前工作目录）
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
echo "[zlib构建器] 项目根目录: $PROJECT_ROOT"

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
    echo "[zlib构建器] 使用默认工具链文件: $TOOLCHAIN_FILE"
fi

# 检查工具链文件是否存在
if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo "警告: 工具链文件不存在: $TOOLCHAIN_FILE"
    echo "       CMake配置可能失败或使用系统默认编译器"
fi

# 设置交叉编译环境变量
export CC=${CROSS_COMPILE_PREFIX}-gcc
export AR=${CROSS_COMPILE_PREFIX}-ar
export RANLIB=${CROSS_COMPILE_PREFIX}-ranlib

echo "[zlib构建器] 开始构建zlib库"
echo "  平台: $PLATFORM"
echo "  安装路径: $INSTALL_DIR/$PLATFORM/zlib"
echo "  工具链文件: $TOOLCHAIN_FILE"

# 检查交叉编译工具链是否可用
echo "[zlib构建器] 检查交叉编译工具链..."
if ! command -v $CC &> /dev/null; then
    echo "警告: 未找到交叉编译工具链 $CC"
    echo "       构建可能失败"
    echo "       x86_64平台请安装: sudo apt install gcc-x86-64-linux-gnu g++-x86-64-linux-gnu"
    echo "       aarch64平台请安装: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
fi

# 下载和构建zlib
mkdir -p ${PROJECT_ROOT}/tmp
mkdir -p ${PROJECT_ROOT}/third_party

cd ${PROJECT_ROOT}/tmp

# 克隆或更新代码
if [ ! -d "zlib" ]; then
    echo "[zlib构建器] 克隆zlib代码..."
    git clone https://github.com/madler/zlib.git
    if [ $? -ne 0 ]; then
        echo "[zlib构建器] 错误: 克隆zlib代码失败"
        exit 1
    fi
else
    echo "[zlib构建器] zlib目录已存在，跳过克隆"
fi

cd zlib

# 清理旧的构建目录
rm -rf build_${PLATFORM}
mkdir -p build_${PLATFORM}
cd build_${PLATFORM}

echo "[zlib构建器] 配置zlib..."
echo "[zlib构建器] 安装前缀: ${INSTALL_DIR}/${PLATFORM}/zlib"
echo "[zlib构建器] 构建静态库: 是"

# 配置zlib
../configure \
    --prefix=${INSTALL_DIR}/${PLATFORM}/zlib \
    --static

if [ $? -ne 0 ]; then
    echo "[zlib构建器] 错误: zlib配置失败"
    exit 1
fi

echo "[zlib构建器] 编译zlib..."
CPU_CORES=$(nproc 2>/dev/null || echo 4)
echo "[zlib构建器] 使用 $CPU_CORES 个CPU核心进行编译"
make -j$CPU_CORES

if [ $? -ne 0 ]; then
    echo "[zlib构建器] 错误: zlib编译失败"
    exit 1
fi

echo "[zlib构建器] 安装zlib..."
make install

if [ $? -ne 0 ]; then
    echo "[zlib构建器] 错误: zlib安装失败"
    exit 1
fi

# 验证安装
if [ -f "${INSTALL_DIR}/${PLATFORM}/zlib/lib/libz.a" ]; then
    echo "[zlib构建器] ✓ zlib静态库安装成功"
    echo "[zlib构建器]   库文件位置: ${INSTALL_DIR}/${PLATFORM}/zlib/lib/libz.a"
    # 额外验证：检查头文件是否存在
    if [ -f "${INSTALL_DIR}/${PLATFORM}/zlib/include/zlib.h" ]; then
        echo "[zlib构建器] ✓ zlib头文件安装成功"
    else
        echo "[zlib构建器] ⚠ 警告: 找不到zlib头文件"
    fi
else
    echo "[zlib构建器] ⚠ 警告: 找不到zlib库文件，但安装命令已成功执行"
    echo "[zlib构建器]   请检查安装目录: ${INSTALL_DIR}/${PLATFORM}/zlib"
    
    # 尝试查找文件
    echo "[zlib构建器]   正在搜索文件..."
    find "${INSTALL_DIR}/${PLATFORM}/zlib" -type f \( -name "*.a" -o -name "*.so" -o -name "*.h" \) 2>/dev/null | head -10
    if [ $? -eq 0 ] && [ $(find "${INSTALL_DIR}/${PLATFORM}/zlib" -type f \( -name "*.a" -o -name "*.so" -o -name "*.h" \) 2>/dev/null | wc -l) -gt 0 ]; then
        echo "[zlib构建器]   找到一些文件，安装可能成功"
    else
        echo "[zlib构建器]   未找到任何文件，安装可能失败"
    fi
fi

echo "[zlib构建器] zlib构建完成"