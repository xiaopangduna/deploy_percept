#!/bin/bash
# 第三方库构建调度脚本
# 用法: bash scripts/build_third_party.sh <platform> --libs <libraries>

set -e

# 显示帮助信息
show_help() {
    echo "第三方库构建系统 - 调度脚本"
    echo ""
    echo "用法:"
    echo "  $0 <platform> --libs <libraries>"
    echo "  $0 --help"
    echo ""
    echo "必需参数:"
    echo "  <platform>                 目标平台 (aarch64, x86_64)"
    echo "  --libs <libraries>         逗号分隔的库列表"
    echo ""
    echo "支持的库:"
    echo "  gtest      - Google Test 测试框架"
    echo "  opencv     - OpenCV 计算机视觉库"
    echo "  spdlog     - 快速C++日志库"
    echo "  rknpu      - Rockchip NPU 库"
    echo "  cnpy       - C++ NumPy 文件读写库"
    echo "  rga        - Rockchip 2D 图形加速库"
    echo ""
    echo "示例:"
    echo "  bash scripts/build_third_party.sh x86_64 --libs gtest,opencv"
    echo "  bash scripts/build_third_party.sh aarch64 --libs spdlog,cnpy"
    echo "  bash scripts/build_third_party.sh x86_64 --libs all"
    echo ""
    echo "目录结构:"
    echo "  scripts/"
    echo "    ├── build_third_party.sh          # 本调度脚本"
    echo "    └── third_party_builders/         # 各个库的构建器脚本"
    echo ""
    echo "注意:"
    echo "  1. 必须在项目根目录下运行，或者通过--project-root指定项目根目录"
    echo "  2. 构建过程中会下载源代码到tmp目录，请确保有足够的磁盘空间"
    echo "  3. 构建的库将安装到third_party/<库名>/<平台>/目录"
    echo ""
}

# 初始化变量
LIBS_TO_BUILD=""
PLATFORM=""

# 特殊处理 --help 参数
# 检查是否有 --help 或 -h 参数
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            show_help
            exit 0
            ;;
    esac
done

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --libs)
            if [ -z "$2" ] || [[ "$2" == -* ]]; then
                echo "错误: --libs 参数需要一个值"
                echo "请使用 $0 --help 查看完整用法"
                exit 1
            fi
            LIBS_TO_BUILD="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            echo "错误: 未知选项: $1"
            echo "请使用 $0 --help 查看完整用法"
            exit 1
            ;;
        *)
            if [ -z "$PLATFORM" ]; then
                PLATFORM="$1"
            else
                echo "错误: 未知参数: $1"
                echo "请使用 $0 --help 查看完整用法"
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查必需参数
if [ -z "$PLATFORM" ]; then
    echo "错误: 必须指定目标平台"
    echo "请使用 $0 --help 查看完整用法"
    exit 1
fi

if [ -z "$LIBS_TO_BUILD" ]; then
    echo "错误: 必须指定要构建的库"
    echo "请使用 $0 --help 查看完整用法"
    exit 1
fi

# 获取项目根目录
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")
echo "项目根目录: $PROJECT_ROOT"

# 第三方构建器目录
THIRD_PARTY_BUILDERS_DIR="$SCRIPT_DIR/third_party_builders"

# 检查第三方构建器目录是否存在
if [ ! -d "$THIRD_PARTY_BUILDERS_DIR" ]; then
    echo "错误: 找不到第三方构建器目录: $THIRD_PARTY_BUILDERS_DIR"
    echo "请确保 scripts/third_party_builders 目录存在"
    exit 1
fi

echo "================================================================"
echo "第三方库构建系统"
echo "================================================================"
echo "平台: $PLATFORM"
echo "要构建的库: $LIBS_TO_BUILD"
echo "构建器目录: $THIRD_PARTY_BUILDERS_DIR"
echo "================================================================"

# 根据平台设置相关变量
case "${PLATFORM}" in
    aarch64)
        TOOLCHAIN_FILE=${PROJECT_ROOT}/cmake/aarch64-toolchain.cmake
        ;;
    x86_64)
        TOOLCHAIN_FILE=${PROJECT_ROOT}/cmake/x86_64-toolchain.cmake
        ;;
    *)
        echo "错误: 不支持的平台 '${PLATFORM}'"
        echo "支持的平台: aarch64, x86_64"
        echo "请使用 $0 --help 查看完整用法"
        exit 1
        ;;
esac

# 创建平台对应的第三方库目录
mkdir -p ${PROJECT_ROOT}/tmp
mkdir -p ${PROJECT_ROOT}/third_party
INSTALL_DIR=${PROJECT_ROOT}/third_party/

echo "安装目录: $INSTALL_DIR"
echo "使用工具链文件: $TOOLCHAIN_FILE"

# 特殊处理 "all" 参数
if [ "$LIBS_TO_BUILD" = "all" ]; then
    echo "检测到 'all' 参数，将构建所有支持的库"
    # 获取所有可用的构建器
    if [ -d "$THIRD_PARTY_BUILDERS_DIR" ]; then
        # 提取所有builder_*.sh文件的库名
        ALL_LIBS=$(ls "$THIRD_PARTY_BUILDERS_DIR"/builder_*.sh 2>/dev/null | 
                   sed 's|.*/builder_||;s|\.sh||' | 
                   tr '\n' ',' | 
                   sed 's/,$//')
        
        if [ -z "$ALL_LIBS" ]; then
            echo "错误: 在 $THIRD_PARTY_BUILDERS_DIR 中找不到任何构建器脚本"
            exit 1
        fi
        
        echo "将构建的库: $ALL_LIBS"
        LIBS_TO_BUILD="$ALL_LIBS"
    else
        echo "错误: 构建器目录不存在: $THIRD_PARTY_BUILDERS_DIR"
        exit 1
    fi
fi

# 解析要构建的库列表
IFS=',' read -ra LIBS_ARRAY <<< "$LIBS_TO_BUILD"

# 遍历所有需要构建的库
for lib in "${LIBS_ARRAY[@]}"; do
    # 去除可能的空格
    lib=$(echo "$lib" | xargs)
    
    # 检查库名是否为空
    if [ -z "$lib" ]; then
        echo "警告: 跳过空的库名"
        continue
    fi
    
    # 检查对应的构建脚本是否存在
    build_script="${THIRD_PARTY_BUILDERS_DIR}/builder_${lib}.sh"
    
    if [ ! -f "$build_script" ]; then
        echo "错误: 库 '$lib' 的构建脚本不存在: $build_script"
        echo "支持的库: gtest, opencv, spdlog, rknpu, cnpy, rga"
        echo "可用构建器:"
        ls -1 "$THIRD_PARTY_BUILDERS_DIR"/builder_*.sh 2>/dev/null | 
            sed 's|.*/builder_||;s|\.sh||' | 
            tr '\n' ' ' | 
            sed 's/ $//'
        echo ""
        echo "请使用 $0 --help 查看完整用法"
        exit 1
    fi
    
    echo ""
    echo "===================================================================="
    echo "开始构建: $lib"
    echo "使用构建器: $(basename "$build_script")"
    echo "===================================================================="
    
    # 调用具体的库构建脚本，传递所需参数
    if ! bash "$build_script" \
        --platform "$PLATFORM" \
        --project-root "$PROJECT_ROOT" \
        --install-dir "$INSTALL_DIR" \
        --toolchain-file "$TOOLCHAIN_FILE"
    then
        echo "错误: 构建 $lib 失败"
        echo "请检查构建日志以获取更多信息"
        exit 1
    fi
    
    echo "完成构建: $lib"
done

echo ""
echo "===================================================================="
echo "第三方库构建完成！"
echo "===================================================================="
echo "所有指定的库已成功构建并安装到:"
echo "$INSTALL_DIR"
echo ""
echo "各个库的安装位置:"
for lib in "${LIBS_ARRAY[@]}"; do
    lib=$(echo "$lib" | xargs)
    if [ -n "$lib" ]; then
        lib_dir="$INSTALL_DIR/$lib/$PLATFORM"
        if [ -d "$lib_dir" ]; then
            echo "  - $lib: $lib_dir"
        fi
    fi
done
echo "===================================================================="