#!/bin/bash
# 第三方库构建器：RKNPU (Rockchip NPU)
# 可以单独运行，也可以由 build_third_party.sh 调用

set -e

# 显示帮助信息
show_help() {
    echo "RKNPU 构建器脚本"
    echo ""
    echo "用途:"
    echo "  获取和安装Rockchip NPU库 (RKNPU1/RKNPU2)"
    echo ""
    echo "用法:"
    echo "  $0 [选项]"
    echo ""
    echo "必需选项:"
    echo "  --platform <平台>          目标平台 (aarch64, x86_64 等)"
    echo ""
    echo "可选选项:"
    echo "  --project-root <路径>      项目根目录 (默认: 当前目录)"
    echo "  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)"
    echo "  --toolchain-file <文件>    CMake工具链文件 (默认: \$PROJECT_ROOT/cmake/\$PLATFORM-toolchain.cmake)"
    echo "  --help                    显示此帮助信息"
    echo ""
    echo "注意:"
    echo "  - RKNPU不是编译的库，而是从Rockchip官方仓库获取预编译库"
    echo "  - RKNPU主要用于Rockchip ARM平台"
    echo "  - x86_64平台会自动跳过（无需使用RKNPU）"
    echo "  - 其他平台将尝试安装，但可能需要自行验证兼容性"
    echo "  - 使用sparse-checkout只下载rknn_model_zoo仓库的3rdparty/rknpu2和3rdparty/rknpu1目录"
    echo "  - --toolchain-file 参数在此脚本中不被使用，仅为接口统一而接受"
    echo ""
    echo "示例:"
    echo "  # 在项目根目录下运行"
    echo "  cd /path/to/project"
    echo "  bash scripts/third_party_builders/builder_rknpu.sh --platform aarch64"
    echo ""
    echo "  # 从任何位置运行"
    echo "  bash /path/to/project/scripts/third_party_builders/builder_rknpu.sh \\"
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
    echo "[RKNPU构建器] 平台为 x86_64，跳过 RKNPU 库的安装（RKNPU 主要用于 ARM 平台）"
    exit 0
fi

echo "[RKNPU构建器] 平台: $PLATFORM"
echo "[RKNPU构建器] 注意: RKNPU 是 Rockchip 平台的预编译库"

# 设置项目根目录默认值（当前工作目录）
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
echo "[RKNPU构建器] 项目根目录: $PROJECT_ROOT"

# 检查项目根目录是否合理
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误: 项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

# 设置安装目录默认值
INSTALL_DIR=${INSTALL_DIR:-${PROJECT_ROOT}/third_party}

# 接受但不使用toolchain-file参数
if [ -n "$TOOLCHAIN_FILE" ]; then
    echo "[RKNPU构建器] 收到工具链文件参数: $TOOLCHAIN_FILE"
    echo "[RKNPU构建器] 注意: RKNPU是预编译库，不需要编译，因此不使用工具链文件"
fi

echo "[RKNPU构建器] 开始处理RKNPU库"
echo "  平台: $PLATFORM"
echo "  安装路径: $INSTALL_DIR"

# 创建临时目录和安装目录
mkdir -p ${PROJECT_ROOT}/tmp
mkdir -p ${INSTALL_DIR}

cd ${PROJECT_ROOT}/tmp

# 克隆或更新rknn_model_zoo仓库（使用sparse-checkout只获取需要的目录）
if [ ! -d "rknn_model_zoo" ]; then
    echo "[RKNPU构建器] 初始化rknn_model_zoo仓库..."
    git init rknn_model_zoo
    cd rknn_model_zoo
    
    echo "[RKNPU构建器] 添加远程仓库..."
    git remote add origin https://github.com/airockchip/rknn_model_zoo.git
    if [ $? -ne 0 ]; then
        echo "[RKNPU构建器] 错误: 添加远程仓库失败"
        exit 1
    fi
    
    echo "[RKNPU构建器] 初始化sparse-checkout..."
    git sparse-checkout init --cone
    if [ $? -ne 0 ]; then
        echo "[RKNPU构建器] 警告: sparse-checkout初始化失败，尝试完整克隆"
        # 如果sparse-checkout失败，回退到完整克隆
        cd ..
        rm -rf rknn_model_zoo
        git clone https://github.com/airockchip/rknn_model_zoo.git
        if [ $? -ne 0 ]; then
            echo "[RKNPU构建器] 错误: 克隆rknn_model_zoo仓库失败"
            exit 1
        fi
        cd rknn_model_zoo
    else
        echo "[RKNPU构建器] 设置要检出的目录..."
        git sparse-checkout set 3rdparty/rknpu2 3rdparty/rknpu1
        if [ $? -ne 0 ]; then
            echo "[RKNPU构建器] 错误: 设置sparse-checkout目录失败"
            exit 1
        fi
        
        echo "[RKNPU构建器] 拉取main分支..."
        git pull origin main
        if [ $? -ne 0 ]; then
            echo "[RKNPU构建器] 错误: 拉取仓库失败"
            exit 1
        fi
    fi
    echo "[RKNPU构建器] 仓库克隆完成"  
else    
    echo "[RGA构建器] librga目录已存在，跳过克隆"
fi

cd ${PROJECT_ROOT}/tmp/rknn_model_zoo
echo "[RKNPU构建器] 开始拷贝RKNPU库文件到third_party目录..."

# 拷贝rknpu2目录
if [ -d "3rdparty/rknpu2" ]; then
    echo "[RKNPU构建器] 发现rknpu2目录，正在拷贝..."
    
    # 如果目标目录已存在，则先删除再拷贝
    if [ -d "${INSTALL_DIR}/rknpu2" ]; then
        echo "[RKNPU构建器] 删除已存在的${INSTALL_DIR}/rknpu2目录..."
        rm -rf ${INSTALL_DIR}/rknpu2
    fi
    
    echo "[RKNPU构建器] 拷贝rknpu2目录到${INSTALL_DIR}/rknpu2..."
    cp -r 3rdparty/rknpu2 ${INSTALL_DIR}/
    if [ $? -ne 0 ]; then
        echo "[RKNPU构建器] 错误: 拷贝rknpu2目录失败"
        exit 1
    fi
    
    # 验证拷贝
    if [ -d "${INSTALL_DIR}/rknpu2" ]; then
        echo "[RKNPU构建器] ✓ rknpu2目录拷贝成功"
        echo "[RKNPU构建器]   rknpu2库文件位置: ${INSTALL_DIR}/rknpu2"
        
        # 检查rknpu2目录结构
        echo "[RKNPU构建器]   rknpu2目录结构:"
        find "${INSTALL_DIR}/rknpu2" -type f -name "*.so" -o -name "*.h" | head -10 | while read file; do
            echo "[RKNPU构建器]     - $(basename "$file")"
        done
    else
        echo "[RKNPU构建器] 错误: 拷贝后找不到rknpu2目录"
        exit 1
    fi
else
    echo "[RKNPU构建器] ⚠ 警告: 未找到3rdparty/rknpu2目录"
fi

# 拷贝rknpu1目录
if [ -d "3rdparty/rknpu1" ]; then
    echo "[RKNPU构建器] 发现rknpu1目录，正在拷贝..."
    
    # 如果目标目录已存在，则先删除再拷贝
    if [ -d "${INSTALL_DIR}/rknpu1" ]; then
        echo "[RKNPU构建器] 删除已存在的${INSTALL_DIR}/rknpu1目录..."
        rm -rf ${INSTALL_DIR}/rknpu1
    fi
    
    echo "[RKNPU构建器] 拷贝rknpu1目录到${INSTALL_DIR}/rknpu1..."
    cp -r 3rdparty/rknpu1 ${INSTALL_DIR}/
    if [ $? -ne 0 ]; then
        echo "[RKNPU构建器] 错误: 拷贝rknpu1目录失败"
        exit 1
    fi
    
    # 验证拷贝
    if [ -d "${INSTALL_DIR}/rknpu1" ]; then
        echo "[RKNPU构建器] ✓ rknpu1目录拷贝成功"
        echo "[RKNPU构建器]   rknpu1库文件位置: ${INSTALL_DIR}/rknpu1"
        
        # 检查rknpu1目录结构
        echo "[RKNPU构建器]   rknpu1目录结构:"
        find "${INSTALL_DIR}/rknpu1" -type f -name "*.so" -o -name "*.h" | head -10 | while read file; do
            echo "[RKNPU构建器]     - $(basename "$file")"
        done
    else
        echo "[RKNPU构建器] 错误: 拷贝后找不到rknpu1目录"
        exit 1
    fi
else
    echo "[RKNPU构建器] ⚠ 警告: 未找到3rdparty/rknpu1目录"
fi

# 检查是否至少有一个目录被成功拷贝
if [ ! -d "${INSTALL_DIR}/rknpu2" ] && [ ! -d "${INSTALL_DIR}/rknpu1" ]; then
    echo "[RKNPU构建器] 错误: 未找到任何RKNPU库目录"
    echo "[RKNPU构建器]   请检查rknn_model_zoo仓库是否包含rknpu1或rknpu2目录"
    exit 1
fi

echo "[RKNPU构建器] RKNPU库处理完成"
echo "[RKNPU构建器] 成功安装到:"
if [ -d "${INSTALL_DIR}/rknpu2" ]; then
    echo "[RKNPU构建器]   - rknpu2: ${INSTALL_DIR}/rknpu2"
fi
if [ -d "${INSTALL_DIR}/rknpu1" ]; then
    echo "[RKNPU构建器]   - rknpu1: ${INSTALL_DIR}/rknpu1"
fi
echo "[RKNPU构建器] 注意: RKNPU是预编译库，无需编译，直接使用即可"
echo "[RKNPU构建器]       兼容性需根据具体平台验证"