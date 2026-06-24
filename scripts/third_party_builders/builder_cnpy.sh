#!/bin/bash
# 第三方库构建器：cnpy
# 可以单独运行，也可以由 third_party_builder.sh 调用

set -e

readonly CNPY_GIT_URL="https://github.com/rogersce/cnpy.git"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""
TOOLCHAIN_FILE=""
TOOLCHAIN_ROOT=""
TOOLCHAIN_PREFIX=""
BUILD_JOBS=""
USE_TOOLCHAIN_CC=1
CROSS_COMPILE_PREFIX=""

log() { echo "[cnpy构建器] $*"; }

show_help() {
    cat <<EOF
cnpy 构建器脚本

用法:
  bash $0 [选项]

必需选项:
  --platform <平台>          目标平台 (aarch64, aarch64-linux-gnu_orange_pi_4_pro_a733, x86_64, armv7l-SSC375)

可选选项:
  --project-root <路径>      项目根目录 (默认: 当前目录)
  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)
  --toolchain-file <文件>    CMake 工具链 (默认: \$PROJECT_ROOT/cmake/toolchains/\$PLATFORM-toolchain.cmake)
  --jobs <N>                 并行编译线程数 (默认: 平台相关)
  --help                     显示此帮助信息

示例:
  bash scripts/third_party_builders/builder_cnpy.sh --platform x86_64
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build-mode)     shift 2 ;;
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
            USE_TOOLCHAIN_CC=0
            ;;
        armv7l-SSC375|aarch64-linux-gnu_orange_pi_4_pro_a733)
            USE_TOOLCHAIN_CC=0
            ;;
        *)
            echo "错误: 不支持的平台 '${PLATFORM}'"
            echo "支持的平台: aarch64, aarch64-linux-gnu_orange_pi_4_pro_a733, x86_64, armv7l-SSC375"
            exit 1
            ;;
    esac

    if [ -z "${BUILD_JOBS}" ]; then
        local nproc
        nproc=$(nproc 2>/dev/null || echo 4)
        if [[ "${PLATFORM}" == armv7l-SSC375 || "${PLATFORM}" == aarch64-linux-gnu_orange_pi_4_pro_a733 ]]; then
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

    init_modules_tmp
}

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
    if [[ "${PLATFORM}" == armv7l-SSC375 || "${PLATFORM}" == aarch64-linux-gnu_orange_pi_4_pro_a733 ]]; then
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
            aarch64)
                log "  请安装: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
                ;;
            armv7l-SSC375|aarch64-linux-gnu_orange_pi_4_pro_a733)
                log "  请挂载工具链: ${TOOLCHAIN_ROOT}"
                ;;
        esac
    fi
}

prepare_cnpy_source() {
    setup_git_safe_directories

    cd "${TMP_MODULES_DIR}"
    if [ ! -d "cnpy" ]; then
        log "克隆 cnpy 源码..."
        git clone "${CNPY_GIT_URL}"
    else
        log "cnpy 目录已存在，跳过克隆"
    fi
}

configure_build_install() {
    local zlib_root="${INSTALL_DIR}/${PLATFORM}/zlib"
    local build_dir="${TMP_MODULES_DIR}/cnpy/build_${PLATFORM}"

    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    cd "${build_dir}"

    log "配置 CMake..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/${PLATFORM}/cnpy" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DBUILD_SHARED_LIBS=OFF \
        -DZLIB_ROOT="${zlib_root}" \
        -DZLIB_LIBRARY="${zlib_root}/lib/libz.a" \
        -DZLIB_INCLUDE_DIR="${zlib_root}/include"

    log "编译 cnpy (-j${BUILD_JOBS})..."
    cmake --build . -j"${BUILD_JOBS}"

    log "安装 cnpy..."
    cmake --install .
}

verify_installation() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/cnpy"
    if [ -f "${install_prefix}/lib/libcnpy.a" ] || \
       [ -f "${install_prefix}/lib64/libcnpy.a" ] || \
       [ -f "${install_prefix}/include/cnpy.h" ]; then
        log "✓ cnpy 已安装到 ${install_prefix}"
        return 0
    fi

    log "⚠ 未找到 cnpy 安装产物，请检查: ${install_prefix}"
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

    log "平台: ${PLATFORM}  线程: ${BUILD_JOBS}"
    log "安装路径: ${INSTALL_DIR}/${PLATFORM}/cnpy"
    log "工具链: ${TOOLCHAIN_FILE}"

    log "构建依赖 zlib..."
    bash "${SCRIPT_DIR}/builder_zlib.sh" \
        --platform "${PLATFORM}" \
        --project-root "${PROJECT_ROOT}" \
        --install-dir "${INSTALL_DIR}" \
        --toolchain-file "${TOOLCHAIN_FILE}" \
        --jobs "${BUILD_JOBS}"

    prepare_cnpy_source
    configure_build_install
    verify_installation

    local duration=$(( $(date +%s) - start_time ))
    log "cnpy 构建完成 (耗时 ${duration}s)"
}

main "$@"
