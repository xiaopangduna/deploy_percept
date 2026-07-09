#!/bin/bash
# 第三方库构建器：OpenCV
# 可以单独运行，也可以由 third_party_builder.sh 调用
#
# 下载失败时查看: tmp/modules/opencv/build_<platform>/CMakeDownloadLog.txt

set -e

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# readonly OPENCV_GIT_URL="https://gitee.com/opencv/opencv.git"
readonly OPENCV_GIT_URL="https://github.com/opencv/opencv.git"
readonly OPENCV_VERSION="4.5.4"

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

log() { echo "[OpenCV构建器] $*"; }

show_help() {
    cat <<EOF
OpenCV 构建器脚本

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

OpenCV 版本: ${OPENCV_VERSION}

示例:
  bash scripts/third_party_builders/builder_opencv.sh --platform x86_64
  bash scripts/third_party_builders/builder_opencv.sh --platform aarch64-linux-gnu_orange_pi_4_pro_a733 --link shared
EOF
}

build_opencv_cmake_args() {
    local shared_flag

    shared_flag=$(cmake_bool_from_link_type "${EFFECTIVE_LINK_TYPE}")

    OPENCV_CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/${PLATFORM}/opencv"
        -DOPENCV_DOWNLOAD_PATH="${OPENCV_CACHE_DIR}"
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}"
        -DBUILD_SHARED_LIBS="${shared_flag}"
        -DBUILD_ZLIB=ON
        -DBUILD_PNG=ON
        -DPNG_LIBRARY=""
        -DPNG_PNG_INCLUDE_DIR=""
    )
}

configure_build_install() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/opencv"
    local build_dir="${TMP_MODULES_DIR}/opencv/build_${PLATFORM}"

    prepare_third_party_build_dir "${install_prefix}" "${build_dir}"

    OPENCV_CACHE_DIR="${TMP_MODULES_DIR}/opencv_cache"
    mkdir -p "${OPENCV_CACHE_DIR}"

    build_opencv_cmake_args
    log "配置 CMake (缓存: ${OPENCV_CACHE_DIR})..."
    cmake "${TMP_MODULES_DIR}/opencv" "${OPENCV_CMAKE_ARGS[@]}"

    log "编译 OpenCV (-j${BUILD_JOBS})..."
    cmake --build . -j"${BUILD_JOBS}"

    log "安装 OpenCV..."
    cmake --install .
}

verify_installation() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/opencv"

    if verify_installed_library "${install_prefix}" "${EFFECTIVE_LINK_TYPE}" libopencv_core; then
        log "✓ OpenCV (${EFFECTIVE_LINK_TYPE}) 已安装到 ${install_prefix}"
        return 0
    fi

    log "⚠ 未找到 libopencv_core (${EFFECTIVE_LINK_TYPE})，请检查: ${install_prefix}"
    return 1
}

main() {
    local start_time
    start_time=$(date +%s)

    parse_third_party_builder_args "$@"
    require_third_party_platform_arg

    LINK_TYPE=$(validate_link_type "${LINK_TYPE}")
    EFFECTIVE_LINK_TYPE=$(resolve_effective_link_type opencv "${LINK_TYPE}")

    configure_third_party_platform
    setup_third_party_paths
    setup_third_party_toolchain_env cmake

    log "平台: ${PLATFORM}  链接: ${EFFECTIVE_LINK_TYPE}  版本: ${OPENCV_VERSION}  线程: ${BUILD_JOBS}"
    log "安装路径: ${INSTALL_DIR}/${PLATFORM}/opencv"
    log "工具链: ${TOOLCHAIN_FILE}"

    prepare_third_party_git_source_at_ref opencv "${OPENCV_GIT_URL}" "${OPENCV_VERSION}" OpenCV
    configure_build_install
    verify_installation

    local duration=$(( $(date +%s) - start_time ))
    log "OpenCV 构建完成 (耗时 ${duration}s)"
}

main "$@"
