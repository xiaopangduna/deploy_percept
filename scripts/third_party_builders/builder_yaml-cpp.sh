#!/bin/bash
# 第三方库构建器：yaml-cpp
# 可以单独运行，也可以由 third_party_builder.sh 调用

set -e

readonly YAML_CPP_GIT_URL="https://github.com/jbeder/yaml-cpp.git"
readonly YAML_CPP_VERSION="0.8.0"

PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""
TOOLCHAIN_FILE=""
TOOLCHAIN_ROOT=""
TOOLCHAIN_PREFIX=""
BUILD_JOBS=""
USE_TOOLCHAIN_CC=1
CROSS_COMPILE_PREFIX=""

log() { echo "[yaml-cpp构建器] $*"; }

show_help() {
    cat <<EOF
yaml-cpp 构建器脚本

用法:
  bash $0 [选项]

必需选项:
  --platform <平台>          目标平台 (aarch64, x86_64, armv7l-SSC375)

可选选项:
  --project-root <路径>      项目根目录 (默认: 当前目录)
  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)
  --toolchain-file <文件>    CMake 工具链 (默认: \$PROJECT_ROOT/cmake/toolchains/\$PLATFORM-toolchain.cmake)
  --jobs <N>                 并行编译线程数 (默认: 平台相关)
  --help                     显示此帮助信息

版本:
  全平台                     ${YAML_CPP_VERSION}

示例:
  bash scripts/third_party_builders/builder_yaml-cpp.sh --platform x86_64
  bash scripts/third_party_builders/builder_yaml-cpp.sh --platform armv7l-SSC375 --jobs 4
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --platform)       PLATFORM="$2"; shift 2 ;;
            --project-root)   PROJECT_ROOT="$2"; shift 2 ;;
            --install-dir)    INSTALL_DIR="$2"; shift 2 ;;
            --toolchain-file) TOOLCHAIN_FILE="$2"; shift 2 ;;
            --jobs)           BUILD_JOBS="$2"; shift 2 ;;
            --help)           show_help; exit 0 ;;
            *)
                echo "错误: 未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

configure_platform() {
    case "${PLATFORM}" in
        aarch64)
            CROSS_COMPILE_PREFIX="aarch64-linux-gnu"
            ;;
        x86_64)
            CROSS_COMPILE_PREFIX="x86_64-linux-gnu"
            ;;
        armv7l-SSC375)
            USE_TOOLCHAIN_CC=0
            ;;
        *)
            echo "错误: 不支持的平台 '${PLATFORM}'"
            echo "支持的平台: aarch64, x86_64, armv7l-SSC375"
            exit 1
            ;;
    esac

    if [ -z "${BUILD_JOBS}" ]; then
        local nproc
        nproc=$(nproc 2>/dev/null || echo 4)
        if [ "${PLATFORM}" = "armv7l-SSC375" ]; then
            BUILD_JOBS=$(( nproc > 4 ? 4 : nproc ))
        else
            BUILD_JOBS=$(( nproc > 8 ? 8 : nproc ))
        fi
    fi
}

setup_paths() {
    PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
    if [ ! -d "${PROJECT_ROOT}" ]; then
        echo "错误: 项目根目录不存在: ${PROJECT_ROOT}"
        exit 1
    fi

    INSTALL_DIR=${INSTALL_DIR:-${PROJECT_ROOT}/third_party}
    if [ -z "${TOOLCHAIN_FILE}" ]; then
        TOOLCHAIN_FILE="${PROJECT_ROOT}/cmake/toolchains/${PLATFORM}-toolchain.cmake"
    fi

    if [ ! -f "${TOOLCHAIN_FILE}" ]; then
        log "警告: 工具链文件不存在: ${TOOLCHAIN_FILE}"
    fi
}

# 从 cmake/toolchains/<platform>-toolchain.cmake 解析 TOOLCHAIN_ROOT / TOOLCHAIN_PREFIX（SSC375 唯一来源）
read_toolchain_vars() {
    local key value line
    if [ ! -f "${TOOLCHAIN_FILE}" ]; then
        log "错误: 工具链文件不存在: ${TOOLCHAIN_FILE}"
        return 1
    fi

    for key in TOOLCHAIN_ROOT TOOLCHAIN_PREFIX; do
        line=$(grep -E "^set\\(${key} " "${TOOLCHAIN_FILE}" | head -1)
        if [ -z "${line}" ]; then
            log "错误: ${TOOLCHAIN_FILE} 中未找到 set(${key} ...)"
            return 1
        fi
        value=$(sed -E 's/^set\([^ ]+ "([^"]+)".*/\1/' <<< "${line}")
        if [ -z "${value}" ] || [ "${value}" = "${line}" ]; then
            log "错误: 无法解析 ${key}（${TOOLCHAIN_FILE}）"
            return 1
        fi
        printf -v "${key}" '%s' "${value}"
    done
}

setup_toolchain_env() {
    if [ "${PLATFORM}" = "armv7l-SSC375" ]; then
        read_toolchain_vars || exit 1
        CROSS_COMPILE_PREFIX="${TOOLCHAIN_PREFIX}"
        export PATH="${TOOLCHAIN_ROOT}/bin:${PATH}"
    fi

    if [ "${USE_TOOLCHAIN_CC}" = "1" ]; then
        export CC="${CROSS_COMPILE_PREFIX}-gcc"
        export CXX="${CROSS_COMPILE_PREFIX}-g++"
    fi

    if [ -z "${CROSS_COMPILE_PREFIX}" ]; then
        return 0
    fi

    if ! command -v "${CROSS_COMPILE_PREFIX}-g++" &>/dev/null; then
        log "警告: 未找到 ${CROSS_COMPILE_PREFIX}-g++"
        case "${PLATFORM}" in
            x86_64)
                log "  请安装: sudo apt install gcc-x86-64-linux-gnu g++-x86-64-linux-gnu"
                ;;
            aarch64)
                log "  请安装: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
                ;;
            armv7l-SSC375)
                log "  请挂载 Sigmastar 工具链: ${TOOLCHAIN_ROOT}"
                ;;
        esac
    fi
}

setup_git_safe_directories() {
    git config --global --add safe.directory "${PROJECT_ROOT}/tmp" 2>/dev/null || true
    git config --global --add safe.directory "${PROJECT_ROOT}/tmp/yaml-cpp" 2>/dev/null || true
    local git_dir
    for git_dir in "${PROJECT_ROOT}/tmp/"*/; do
        if [ -d "${git_dir}.git" ]; then
            git config --global --add safe.directory "${git_dir}" 2>/dev/null || true
        fi
    done
}

prepare_yaml_cpp_source() {
    mkdir -p "${PROJECT_ROOT}/tmp" "${PROJECT_ROOT}/third_party"
    setup_git_safe_directories

    cd "${PROJECT_ROOT}/tmp"
    if [ ! -d "yaml-cpp" ]; then
        log "克隆 yaml-cpp 源码..."
        git clone "${YAML_CPP_GIT_URL}"
    else
        log "yaml-cpp 目录已存在，跳过克隆"
    fi

    cd yaml-cpp
    log "切换到版本 ${YAML_CPP_VERSION}..."
    git checkout "${YAML_CPP_VERSION}"
}

build_yaml_cpp_cmake_args() {
    YAML_CPP_CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/${PLATFORM}/yaml-cpp"
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}"
        -DYAML_CPP_BUILD_TESTS=OFF
        -DYAML_CPP_BUILD_TOOLS=OFF
        -DYAML_BUILD_SHARED_LIBS=OFF
    )
}

configure_build_install() {
    local build_dir="build_${PLATFORM}"
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    cd "${build_dir}"

    build_yaml_cpp_cmake_args
    log "配置 CMake..."
    cmake .. "${YAML_CPP_CMAKE_ARGS[@]}"

    log "编译 yaml-cpp (-j${BUILD_JOBS})..."
    make -j"${BUILD_JOBS}"

    log "安装 yaml-cpp..."
    make install
}

verify_installation() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/yaml-cpp"
    local lib_dir=""

    if [ -f "${install_prefix}/lib/libyaml-cpp.a" ]; then
        lib_dir="${install_prefix}/lib"
    elif [ -f "${install_prefix}/lib64/libyaml-cpp.a" ]; then
        lib_dir="${install_prefix}/lib64"
    elif [ -f "${install_prefix}/lib/libyaml-cpp.so" ]; then
        lib_dir="${install_prefix}/lib"
    elif [ -f "${install_prefix}/lib64/libyaml-cpp.so" ]; then
        lib_dir="${install_prefix}/lib64"
    fi

    if [ -n "${lib_dir}" ]; then
        log "✓ yaml-cpp 已安装到 ${install_prefix}"
        log "  库文件: ${lib_dir}/libyaml-cpp.a"
        if [ -f "${install_prefix}/lib/cmake/yaml-cpp/yaml-cpp-config.cmake" ]; then
            log "  CMake 配置: ${install_prefix}/lib/cmake/yaml-cpp/yaml-cpp-config.cmake"
        fi
        return 0
    fi

    if [ -f "${install_prefix}/include/yaml-cpp/yaml.h" ]; then
        log "✓ yaml-cpp 头文件已安装: ${install_prefix}"
        return 0
    fi

    log "⚠ 未找到 libyaml-cpp 或头文件，请检查: ${install_prefix}"
    find "${install_prefix}" -type f \( -name "*.a" -o -name "*.so" -o -name "yaml.h" \) 2>/dev/null | head -10 || true
    return 1
}

main() {
    local start_time
    start_time=$(date +%s)

    parse_args "$@"

    if [ -z "${PLATFORM}" ]; then
        echo "错误: 必须指定平台 (--platform)"
        show_help
        exit 1
    fi

    configure_platform
    setup_paths
    setup_toolchain_env

    log "平台: ${PLATFORM}  版本: ${YAML_CPP_VERSION}  线程: ${BUILD_JOBS}"
    if [ -n "${CROSS_COMPILE_PREFIX}" ]; then
        log "交叉编译前缀: ${CROSS_COMPILE_PREFIX}"
    fi
    log "安装路径: ${INSTALL_DIR}/${PLATFORM}/yaml-cpp"
    log "工具链: ${TOOLCHAIN_FILE}"

    prepare_yaml_cpp_source
    configure_build_install
    verify_installation

    local duration=$(( $(date +%s) - start_time ))
    log "yaml-cpp 构建完成 (耗时 ${duration}s)"
}

main "$@"
