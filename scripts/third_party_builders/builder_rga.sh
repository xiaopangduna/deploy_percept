#!/bin/bash
# 第三方库构建器：RGA (Rockchip Graphics Acceleration)
# 可以单独运行，也可以由 third_party_builder.sh 调用
# 注意：RGA 是预编译库，此脚本仅负责获取源代码，不进行编译

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 显示帮助信息
show_help() {
    echo "RGA 构建器脚本"
    echo ""
    echo "用途:"
    echo "  获取并安装 Rockchip RGA (librga) 源代码"
    echo ""
    echo "用法:"
    echo "  $0 [选项]"
    echo ""
    echo "必需选项:"
    echo "  --platform <平台>          目标平台 (aarch64, x86_64) [必需]"
    echo ""
    echo "可选选项:"
    echo "  --build-mode <host|cross>   构建模式（接口统一，实际不使用）"
    echo "  --project-root <路径>      项目根目录 (默认: 当前目录)"
    echo "  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)"
    echo "  --toolchain-file <文件>    CMake工具链文件（当前不使用，仅为接口统一）"
    echo "  --help                    显示此帮助信息"
    echo ""
    echo "注意:"
    echo "  - RGA 主要用于 Rockchip ARM 平台"
    echo "  - 当前脚本仅拷贝 librga 源代码，不进行编译"
    echo "  - x86_64 平台会自动跳过安装（视为成功）"
    echo "  - --build-mode 参数仅为接口统一而保留，实际不使用"
    echo ""
    echo "示例:"
    echo "  bash scripts/third_party_builders/builder_rga.sh --platform aarch64"
}

# -------------------------
# 参数解析
# -------------------------
BUILD_MODE=""  # 仅为接口统一而保留
PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""
TOOLCHAIN_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-mode)
            BUILD_MODE="$2"
            echo "[RGA构建器] 警告: --build-mode 参数不被使用，仅为接口统一"
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
            echo "错误: 未知参数 $1"
            show_help
            exit 1
            ;;
    esac
done

# -------------------------
# 基本检查
# -------------------------
if [ -z "$PLATFORM" ]; then
    echo "错误: 必须指定 --platform"
    show_help
    exit 1
fi

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
INSTALL_DIR=${INSTALL_DIR:-${PROJECT_ROOT}/third_party}

echo "[RGA构建器] 平台: $PLATFORM"
echo "[RGA构建器] 项目根目录: $PROJECT_ROOT"
echo "[RGA构建器] 安装根目录: $INSTALL_DIR"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误: 项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

# toolchain-file 仅接受，不使用
if [ -n "$TOOLCHAIN_FILE" ]; then
    echo "[RGA构建器] 收到 toolchain-file 参数（当前不使用）: $TOOLCHAIN_FILE"
fi

# -------------------------
# 平台策略
# -------------------------
if [ "$PLATFORM" = "x86_64" ]; then
    echo "[RGA构建器] x86_64 平台不支持 RGA，跳过安装"
    echo "[RGA构建器] 视为成功完成"
    exit 0
fi

if [ "$PLATFORM" != "aarch64" ]; then
    echo "错误: RGA 当前仅支持 aarch64 平台"
    exit 1
fi

# -------------------------
# 准备目录
# -------------------------
TMP_DIR="${PROJECT_ROOT}/tmp"
RGA_INSTALL_DIR="${INSTALL_DIR}/rga/${PLATFORM}"

mkdir -p "$TMP_DIR"
mkdir -p "$INSTALL_DIR"

cd "$TMP_DIR"

# -------------------------
# 获取 librga
# -------------------------
if [ ! -d "librga" ]; then
    echo "[RGA构建器] 克隆 librga 仓库..."
    git clone https://github.com/airockchip/librga.git
else
    echo "[RGA构建器] librga 仓库已存在，跳过克隆"
fi

# -------------------------
# 安装（拷贝源码）
# -------------------------
echo "[RGA构建器] 安装路径: $RGA_INSTALL_DIR"

if [ -d "$RGA_INSTALL_DIR" ]; then
    echo "[RGA构建器] 清理已存在的目录: $RGA_INSTALL_DIR"
    rm -rf "$RGA_INSTALL_DIR"
fi

mkdir -p "$RGA_INSTALL_DIR"
cp -r "$TMP_DIR/librga" "$RGA_INSTALL_DIR/"

# -------------------------
# 验证
# -------------------------
if [ -d "$RGA_INSTALL_DIR/librga" ]; then
    echo "[RGA构建器] ✓ RGA 源码安装完成"
    echo "[RGA构建器]   位置: $RGA_INSTALL_DIR/librga"

    echo "[RGA构建器]   目录内容预览:"
    find "$RGA_INSTALL_DIR/librga" -maxdepth 2 -type f \
        \( -name "*.h" -o -name "*.c" -o -name "*.cpp" -o -name "CMakeLists.txt" \) \
        | head -10 | while read f; do
            echo "     - $(basename "$f")"
        done
else
    echo "[RGA构建器] 错误: RGA 安装失败，目录不存在"
    exit 1
fi

echo "[RGA构建器] RGA 处理完成"
echo "[RGA构建器] 说明: 当前仅拷贝源码，如需使用请根据平台自行编译"
