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

THIRD_PARTY_IGNORE_LINK_TYPE=1

log() { echo "[cnpy构建器] $*"; }

show_help() {
    cat <<EOF
cnpy 构建器脚本

用法:
  bash $0 [选项]

必需选项:
  --platform <平台>          与 cmake/toolchains/<platform>-toolchain.cmake 同名

可选选项:
  --project-root <路径>      项目根目录 (默认: 当前目录)
  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)
  --toolchain-file <文件>    CMake 工具链 (默认: \$PROJECT_ROOT/cmake/toolchains/\$PLATFORM-toolchain.cmake)
  --jobs <N>                 并行编译线程数 (默认: min(nproc, 8))
  --help                     显示此帮助信息

注意:
  - cnpy 上游默认同时安装 libcnpy.a 与 libcnpy.so（不受 --link 影响）
  - 主工程 FindCnpyCustom 默认使用 libcnpy.a

示例:
  bash scripts/third_party_builders/builder_cnpy.sh --platform x86_64
EOF
}

configure_build_install() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/cnpy"
    local zlib_root="${INSTALL_DIR}/${PLATFORM}/zlib"
    local build_dir="${TMP_MODULES_DIR}/cnpy/build_${PLATFORM}"

    prepare_third_party_build_dir "${install_prefix}" "${build_dir}"

    log "配置 CMake（上游默认：libcnpy.a + libcnpy.so）..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
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
    if [ -f "${install_prefix}/lib/libcnpy.a" ] && \
       [ -f "${install_prefix}/include/cnpy.h" ]; then
        log "✓ cnpy 已安装到 ${install_prefix}"
        if [ -f "${install_prefix}/lib/libcnpy.so" ]; then
            log "  含 libcnpy.a（主工程使用）与 libcnpy.so（上游默认）"
        fi
        return 0
    fi

    log "⚠ 未找到 libcnpy.a 或 cnpy.h，请检查: ${install_prefix}"
    return 1
}

main() {
    local start_time
    start_time=$(date +%s)

    parse_third_party_builder_args "$@"
    require_third_party_platform_arg

    configure_third_party_platform
    setup_third_party_paths
    setup_third_party_toolchain_env cmake

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

    prepare_third_party_git_source cnpy "${CNPY_GIT_URL}" cnpy
    configure_build_install
    verify_installation

    local duration=$(( $(date +%s) - start_time ))
    log "cnpy 构建完成 (耗时 ${duration}s)"
}

main "$@"
