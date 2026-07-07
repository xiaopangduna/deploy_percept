#!/bin/bash
# 第三方库构建器：spdlog
# 可以单独运行，也可以由 third_party_builder.sh 调用

set -e

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# readonly SPDLOG_GIT_URL="https://gitee.com/mirror-luyi/spdlog.git"
readonly SPDLOG_GIT_URL="https://github.com/gabime/spdlog.git"
readonly SPDLOG_VERSION="v1.14.1"

PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""
TOOLCHAIN_FILE=""
TOOLCHAIN_ROOT=""
TOOLCHAIN_PREFIX=""
BUILD_JOBS=""
USE_TOOLCHAIN_CC=1
CROSS_COMPILE_PREFIX=""
LINK_TYPE=""
EFFECTIVE_LINK_TYPE=""

log() { echo "[spdlog构建器] $*"; }

show_help() {
    cat <<EOF
spdlog 构建器脚本

用法:
  bash $0 [选项]

必需选项:
  --platform <平台>          与 cmake/toolchains/<platform>-toolchain.cmake 同名

可选选项:
  --project-root <路径>      项目根目录 (默认: 当前目录)
  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)
  --toolchain-file <文件>    CMake 工具链 (默认: \$PROJECT_ROOT/cmake/toolchains/\$PLATFORM-toolchain.cmake)
  --link-type static|shared  链接类型（默认 static）
  --jobs <N>                 并行编译线程数 (默认: min(nproc, 8))
  --help                     显示此帮助信息

spdlog 版本: ${SPDLOG_VERSION}

示例:
  bash scripts/third_party_builders/builder_spdlog.sh --platform x86_64
  bash scripts/third_party_builders/builder_spdlog.sh --platform aarch64-linux-gnu_orange_pi_4_pro_a733 --link shared
EOF
}

build_spdlog_cmake_args() {
    local shared_flag
    shared_flag=$(cmake_bool_from_link_type "${EFFECTIVE_LINK_TYPE}")

    SPDLOG_CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/${PLATFORM}/spdlog"
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}"
        -DSPDLOG_BUILD_SHARED="${shared_flag}"
        -DSPDLOG_BUILD_EXAMPLE=OFF
        -DSPDLOG_BUILD_BENCH=OFF
    )

    if [[ "${PLATFORM}" == aarch64-linux-gnu_orange_pi_4_pro_a733 ]]; then
        SPDLOG_CMAKE_ARGS+=(
            -DSPDLOG_ENABLE_PCH=OFF
        )
    fi
}

configure_build_install() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/spdlog"
    local build_dir="${TMP_MODULES_DIR}/spdlog/build_${PLATFORM}"

    prepare_third_party_build_dir "${install_prefix}" "${build_dir}"

    build_spdlog_cmake_args
    log "配置 CMake (${EFFECTIVE_LINK_TYPE})..."
    cmake "${TMP_MODULES_DIR}/spdlog" "${SPDLOG_CMAKE_ARGS[@]}"

    log "编译 spdlog (-j${BUILD_JOBS})..."
    cmake --build . -j"${BUILD_JOBS}"

    log "安装 spdlog..."
    cmake --install .
}

verify_installation() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/spdlog"

    if verify_installed_library "${install_prefix}" "${EFFECTIVE_LINK_TYPE}" libspdlog; then
        log "✓ spdlog (${EFFECTIVE_LINK_TYPE}) 已安装到 ${install_prefix}"
        if [ -f "${install_prefix}/lib/cmake/spdlog/spdlogConfig.cmake" ]; then
            log "  CMake 配置: ${install_prefix}/lib/cmake/spdlog/spdlogConfig.cmake"
        fi
        return 0
    fi

    if [ -f "${install_prefix}/include/spdlog/spdlog.h" ]; then
        log "✓ spdlog 头文件已安装（header-only 模式）: ${install_prefix}"
        return 0
    fi

    log "⚠ 未找到 libspdlog (${EFFECTIVE_LINK_TYPE}) 或头文件，请检查: ${install_prefix}"
    return 1
}

main() {
    local start_time
    start_time=$(date +%s)

    parse_third_party_builder_args "$@"
    require_third_party_platform_arg

    LINK_TYPE=$(validate_link_type "${LINK_TYPE}")
    EFFECTIVE_LINK_TYPE=$(resolve_effective_link_type spdlog "${LINK_TYPE}")

    configure_third_party_platform
    setup_third_party_paths
    setup_third_party_toolchain_env cmake

    log "平台: ${PLATFORM}  链接: ${EFFECTIVE_LINK_TYPE}  版本: ${SPDLOG_VERSION}  线程: ${BUILD_JOBS}"
    log "安装路径: ${INSTALL_DIR}/${PLATFORM}/spdlog"
    log "工具链: ${TOOLCHAIN_FILE}"

    prepare_third_party_git_source_at_ref spdlog "${SPDLOG_GIT_URL}" "${SPDLOG_VERSION}" spdlog
    configure_build_install
    verify_installation

    local duration=$(( $(date +%s) - start_time ))
    log "spdlog 构建完成 (耗时 ${duration}s)"
}

main "$@"
