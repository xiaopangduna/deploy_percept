#!/bin/bash
# 第三方库构建器：RGA (Rockchip Raster Graphic Acceleration)
# 可以单独运行，也可以由 build_third_party.sh 调用

set -e

# 显示帮助信息
show_help() {
    echo "RGA 构建器脚本"
    echo ""
    echo "用途:"
    echo "  获取和安装Rockchip RGA库 (Raster Graphic Acceleration)"
    echo ""
    echo "用法:"
    echo "  $0 [选项]"
    echo ""
    echo "必需选项:"
    echo "  --platform <平台>          目标平台 (必须是 aarch64) [必需]"
    echo ""
    echo "可选选项:"
    echo "  --project-root <路径>      项目根目录 (默认: 当前目录)"
    echo "  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)"
    echo "  --toolchain-file <文件>    CMake工具链文件 (默认: \$PROJECT_ROOT/cmake/\$PLATFORM-toolchain.cmake)"
    echo "  --help                    显示此帮助信息"
    echo ""
    echo "注意:"
    echo "  - RGA是Rockchip的2D图形加速库"
    echo "  - 该脚本克隆Rockchip官方的librga仓库并拷贝到安装目录"
    echo "  - RGA通常需要配合Rockchip平台使用"
    echo ""
    echo "示例:"
    echo "  # 在项目根目录下运行"
    echo "  cd /path/to/project"
    echo "  bash scripts/third_party_builders/builder_rga.sh --platform aarch64"
    echo ""
    echo "  # 从任何位置运行"
    echo "  bash /path/to/project/scripts/third_party_builders/builder_rga.sh \\"
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

# 检查平台：x86_64跳过，其他平台正常安装
if [ "$PLATFORM" = "x86_64" ]; then
    echo "[RGA构建器] 平台为 x86_64，跳过 RKNPU 库的安装（RGA 主要用于 ARM 平台）"
    exit 0
fi

echo "[RGA构建器] 平台: $PLATFORM"

# 设置项目根目录默认值（当前工作目录）
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
echo "[RGA构建器] 项目根目录: $PROJECT_ROOT"

# 检查项目根目录是否合理
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误: 项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

# 设置安装目录默认值
INSTALL_DIR=${INSTALL_DIR:-${PROJECT_ROOT}/third_party}

# 接受但不使用toolchain-file参数
if [ -n "$TOOLCHAIN_FILE" ]; then
    echo "[RGA构建器] 收到工具链文件参数: $TOOLCHAIN_FILE"
    echo "[RGA构建器] 注意: RGA库可能包含源代码，目前直接拷贝整个仓库"
fi

echo "[RGA构建器] 开始处理RGA库"
echo "  平台: $PLATFORM"
echo "  安装路径: $INSTALL_DIR/rga"

# 创建临时目录和安装目录
mkdir -p ${PROJECT_ROOT}/tmp
mkdir -p ${INSTALL_DIR}

cd ${PROJECT_ROOT}/tmp

# 克隆或更新librga仓库
if [ ! -d "librga" ]; then
    echo "[RGA构建器] 克隆librga仓库..."
    git clone https://github.com/airockchip/librga.git
    if [ $? -ne 0 ]; then
        echo "[RGA构建器] 错误: 克隆librga仓库失败"
        echo "      请检查网络连接或git配置"
        exit 1
    fi
else
    echo "[RGA构建器] librga目录已存在，跳过克隆"

fi

cd ${PROJECT_ROOT}/tmp/librga
echo "[RGA构建器] 开始拷贝librga库文件到third_party目录..."

# 如果目标目录已存在，则先删除再拷贝
if [ -d "${INSTALL_DIR}/rga" ]; then
    echo "[RGA构建器] 删除已存在的${INSTALL_DIR}/rga目录..."
    rm -rf ${INSTALL_DIR}/rga
fi

# 拷贝整个librga目录到third_party下的rga目录
echo "[RGA构建器] 拷贝librga目录到${INSTALL_DIR}/rga..."
cp -r . ${INSTALL_DIR}/rga/
if [ $? -ne 0 ]; then
    echo "[RGA构建器] 错误: 拷贝librga目录失败"
    exit 1
fi

# 验证拷贝
if [ -d "${INSTALL_DIR}/rga" ]; then
    echo "[RGA构建器] ✓ librga目录拷贝成功"
    echo "[RGA构建器]   RGA库文件位置: ${INSTALL_DIR}/rga"
    
    # 检查RGA目录结构
    echo "[RGA构建器]   RGA目录结构概览:"
    
    # 查找可能的库文件
    find "${INSTALL_DIR}/rga" -type f \( -name "*.so" -o -name "*.a" -o -name "*.h" \) 2>/dev/null | head -10 | while read file; do
        echo "[RGA构建器]     - $(basename "$file")"
    done
    
    # 检查是否有CMakeLists.txt或其他构建文件
    if [ -f "${INSTALL_DIR}/rga/CMakeLists.txt" ]; then
        echo "[RGA构建器]   ✓ 找到CMakeLists.txt文件，可能需要自行编译"
    fi
    
    # 检查是否有include目录
    if [ -d "${INSTALL_DIR}/rga/include" ]; then
        echo "[RGA构建器]   ✓ 找到include目录"
    fi
    
    # 检查是否有lib目录
    if [ -d "${INSTALL_DIR}/rga/lib" ]; then
        echo "[RGA构建器]   ✓ 找到lib目录"
    fi
else
    echo "[RGA构建器] 错误: 拷贝后找不到rga目录"
    exit 1
fi

echo "[RGA构建器] RGA库处理完成"
echo "[RGA构建器] 成功安装到: ${INSTALL_DIR}/rga"
echo "[RGA构建器] 注意: RGA库可能需要根据目标平台进行编译"
echo "[RGA构建器]       当前只拷贝了源代码，请根据需要自行编译"