#!/bin/bash
# 第三方库构建器：zlib
# 可以单独运行，也可以由 third_party_builder.sh 调用

set -e

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

readonly ZLIB_GIT_URL="https://github.com/madler/zlib.git"

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

log() { echo "[zlib构建器] $*"; }

show_help() {
    cat <<EOF
zlib 构建器脚本

用法:
  bash $0 [选项]

必需选项:
  --platform <平台>          与 cmake/toolchains/<platform>-toolchain.cmake 同名

可选选项:
  --project-root <路径>      项目根目录 (默认: 当前目录)
  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)
  --toolchain-file <文件>    接口统一参数；zlib 使用 configure，嵌入式平台通过 CC/AR 交叉编译
  --jobs <N>                 并行编译线程数 (默认: min(nproc, 8))
  --help                     显示此帮助信息

注意:
  - zlib 使用 configure/make，不使用 CMake
  - zlib 上游 configure 默认同时安装 libz.a 与 libz.so（不受 --link 影响）
  - 主工程 FindZlibCustom 默认使用 libz.a

示例:
  bash scripts/third_party_builders/builder_zlib.sh --platform x86_64
EOF
}

configure_build_install() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/zlib"
    local build_dir="${TMP_MODULES_DIR}/zlib/build_${PLATFORM}"

    prepare_third_party_build_dir "${install_prefix}" "${build_dir}"

    log "配置 zlib（上游默认：libz.a + libz.so）..."
    "${TMP_MODULES_DIR}/zlib/configure" \
        --prefix="${install_prefix}"

    log "编译 zlib (-j${BUILD_JOBS})..."
    make -j"${BUILD_JOBS}"

    log "安装 zlib..."
    make install
}

verify_installation() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/zlib"
    if [ -f "${install_prefix}/lib/libz.a" ] && \
       [ -f "${install_prefix}/include/zlib.h" ]; then
        log "✓ zlib 已安装到 ${install_prefix}"
        if [ -f "${install_prefix}/lib/libz.so" ]; then
            log "  含 libz.a（主工程默认）与 libz.so（上游默认）"
        else
            log "  警告: 未找到 libz.so，configure 可能未生成 shared 库"
        fi
        return 0
    fi

    log "⚠ 未找到 libz.a 或 zlib.h，请检查: ${install_prefix}"
    return 1
}

main() {
    local start_time
    start_time=$(date +%s)

    parse_third_party_builder_args "$@"
    require_third_party_platform_arg

    configure_third_party_platform
    setup_third_party_paths
    setup_third_party_toolchain_env configure

    log "平台: ${PLATFORM}  线程: ${BUILD_JOBS}"
    log "安装路径: ${INSTALL_DIR}/${PLATFORM}/zlib"

    prepare_third_party_git_source zlib "${ZLIB_GIT_URL}" zlib
    configure_build_install
    verify_installation

    local duration=$(( $(date +%s) - start_time ))
    log "zlib 构建完成 (耗时 ${duration}s)"
}

main "$@"
