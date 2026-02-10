#!/bin/bash
# 第三方库构建器：RKNPU (Rockchip NPU)
# 可以单独运行，也可以由 build_third_party.sh 调用

set -e

# ------------------------------------------------------------------------------
# 脚本自身目录（保证任何调用方式下路径都可靠）
# ------------------------------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ------------------------------------------------------------------------------
# 显示帮助信息
# ------------------------------------------------------------------------------
show_help() {
    echo "RKNPU 构建器脚本"
    echo ""
    echo "用途:"
    echo "  获取和安装 Rockchip NPU 库 (RKNPU1 / RKNPU2)"
    echo ""
    echo "用法:"
    echo "  $0 [选项]"
    echo ""
    echo "必需选项:"
    echo "  --platform <平台>          目标平台 (aarch64, x86_64 等)"
    echo ""
    echo "可选选项:"
    echo "  --project-root <路径>      项目根目录"
    echo "                            (默认: scripts/third_party_builders 的上两级)"
    echo "  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)"
    echo "  --toolchain-file <文件>    CMake工具链文件（接口统一，实际不使用）"
    echo "  --help                    显示此帮助信息"
    echo ""
    echo "注意:"
    echo "  - RKNPU 是 Rockchip 官方提供的预编译库，不进行编译"
    echo "  - 主要用于 Rockchip ARM 平台"
    echo "  - x86_64 平台将自动跳过"
    echo "  - 使用 sparse-checkout 仅下载 rknpu1 / rknpu2 目录"
    echo "  - --build-mode 参数仅为接口统一而保留，实际不使用"
    echo ""
    echo "示例:"
    echo "  # 在项目根目录下运行"
    echo "  bash scripts/third_party_builders/builder_rknpu.sh --platform aarch64"
    echo ""
    echo "  # 从任意位置运行"
    echo "  bash /path/to/project/scripts/third_party_builders/builder_rknpu.sh \\"
    echo "    --platform aarch64 \\"
    echo "    --project-root /path/to/project"
}

# ------------------------------------------------------------------------------
# 参数初始化
# ------------------------------------------------------------------------------
BUILD_MODE=""  # 仅为接口统一而保留
PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""
TOOLCHAIN_FILE=""

# ------------------------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-mode)
            BUILD_MODE="$2"
            echo "[RKNPU构建器] 警告: --build-mode 参数不被使用，仅为接口统一"
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

# ------------------------------------------------------------------------------
# 参数检查
# ------------------------------------------------------------------------------
if [ -z "$PLATFORM" ]; then
    echo "错误: 必须指定 --platform"
    show_help
    exit 1
fi

# ------------------------------------------------------------------------------
# 平台处理逻辑
# ------------------------------------------------------------------------------
if [ "$PLATFORM" = "x86_64" ]; then
    echo "[RKNPU构建器] 平台为 x86_64，跳过 RKNPU 安装（仅适用于 Rockchip ARM）"
    exit 0
fi

echo "[RKNPU构建器] 平台: $PLATFORM"
echo "[RKNPU构建器] RKNPU 为 Rockchip 平台预编译库"

# ------------------------------------------------------------------------------
# 项目根目录（默认推导）
# ------------------------------------------------------------------------------
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
echo "[RKNPU构建器] 项目根目录: $PROJECT_ROOT"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误: 项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

# ------------------------------------------------------------------------------
# 安装目录
# ------------------------------------------------------------------------------
INSTALL_DIR=${INSTALL_DIR:-${PROJECT_ROOT}/third_party}
echo "[RKNPU构建器] 安装目录: $INSTALL_DIR"

# ------------------------------------------------------------------------------
# toolchain-file 参数说明（接口统一）
# ------------------------------------------------------------------------------
if [ -n "$TOOLCHAIN_FILE" ]; then
    echo "[RKNPU构建器] 收到 --toolchain-file: $TOOLCHAIN_FILE"
    echo "[RKNPU构建器] 说明: RKNPU 为预编译库，该参数不会被使用"
fi

# ------------------------------------------------------------------------------
# 准备临时目录
# ------------------------------------------------------------------------------
TMP_DIR="${PROJECT_ROOT}/tmp"
mkdir -p "$TMP_DIR" "$INSTALL_DIR"
cd "$TMP_DIR"

# ------------------------------------------------------------------------------
# 获取 rknn_model_zoo（sparse-checkout）
# ------------------------------------------------------------------------------
if [ ! -d "rknn_model_zoo" ]; then
    echo "[RKNPU构建器] 初始化 rknn_model_zoo 仓库..."

    git init rknn_model_zoo
    cd rknn_model_zoo

    git remote add origin https://github.com/airockchip/rknn_model_zoo.git

    if git sparse-checkout init --cone; then
        git sparse-checkout set 3rdparty/rknpu2 3rdparty/rknpu1
        git pull origin main
    else
        echo "[RKNPU构建器] sparse-checkout 失败，回退为完整克隆"
        cd ..
        rm -rf rknn_model_zoo
        git clone https://github.com/airockchip/rknn_model_zoo.git
        cd rknn_model_zoo
    fi
else
    echo "[RKNPU构建器] rknn_model_zoo 已存在，跳过获取"
    cd rknn_model_zoo
fi

# ------------------------------------------------------------------------------
# 拷贝 rknpu2
# ------------------------------------------------------------------------------
if [ -d "3rdparty/rknpu2" ]; then
    echo "[RKNPU构建器] 安装 rknpu2 ..."
    rm -rf "${INSTALL_DIR}/rknpu2"
    cp -r 3rdparty/rknpu2 "${INSTALL_DIR}/"
    echo "[RKNPU构建器] ✓ rknpu2 已安装"
else
    echo "[RKNPU构建器] ⚠ 未找到 rknpu2"
fi

# ------------------------------------------------------------------------------
# 拷贝 rknpu1
# ------------------------------------------------------------------------------
if [ -d "3rdparty/rknpu1" ]; then
    echo "[RKNPU构建器] 安装 rknpu1 ..."
    rm -rf "${INSTALL_DIR}/rknpu1"
    cp -r 3rdparty/rknpu1 "${INSTALL_DIR}/"
    echo "[RKNPU构建器] ✓ rknpu1 已安装"
else
    echo "[RKNPU构建器] ⚠ 未找到 rknpu1"
fi

# ------------------------------------------------------------------------------
# 最终检查
# ------------------------------------------------------------------------------
if [ ! -d "${INSTALL_DIR}/rknpu1" ] && [ ! -d "${INSTALL_DIR}/rknpu2" ]; then
    echo "[RKNPU构建器] 错误: 未成功安装任何 RKNPU 组件"
    exit 1
fi

echo "[RKNPU构建器] 完成"
echo "[RKNPU构建器] 注意: RKNPU 为预编译库，请自行验证平台兼容性"
