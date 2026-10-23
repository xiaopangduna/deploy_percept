#!/bin/bash
# 第三方库构建器：nlohmann/json（header-only）
# 可以单独运行，也可以由 third_party_builder.sh 调用

set -e

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

readonly NLOHMANN_JSON_GIT_URL="https://github.com/nlohmann/json.git"
readonly NLOHMANN_JSON_VERSION="v3.11.3"

PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""
TOOLCHAIN_FILE=""
TOOLCHAIN_ROOT=""
TOOLCHAIN_PREFIX=""
BUILD_JOBS=""
USE_TOOLCHAIN_CC=1
CROSS_COMPILE_PREFIX=""

# header-only：忽略 --link-type
THIRD_PARTY_IGNORE_LINK_TYPE=1

log() { echo "[nlohmann_json构建器] $*"; }

show_help() {
    cat <<EOF
nlohmann/json 构建器脚本（header-only）

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

说明:
  本库为 header-only，--link-type 无效；安装头文件与 CMake config 即可。

nlohmann/json 版本: ${NLOHMANN_JSON_VERSION}

示例:
  bash scripts/third_party_builders/builder_nlohmann_json.sh --platform x86_64
  bash scripts/third_party_builders/builder_nlohmann_json.sh --platform armv7l-linux-gnueabihf_ssc308qe
EOF
}

configure_build_install() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/nlohmann_json"
    local build_dir="${TMP_MODULES_DIR}/nlohmann_json/build_${PLATFORM}"

    prepare_third_party_build_dir "${install_prefix}" "${build_dir}"

    log "配置 CMake (header-only)..."
    cmake "${TMP_MODULES_DIR}/nlohmann_json" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DJSON_BuildTests=OFF \
        -DJSON_Install=ON

    log "编译 nlohmann_json (-j${BUILD_JOBS})..."
    cmake --build . -j"${BUILD_JOBS}"

    log "安装 nlohmann_json..."
    cmake --install .
}

verify_installation() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/nlohmann_json"

    if [ -f "${install_prefix}/include/nlohmann/json.hpp" ]; then
        log "✓ nlohmann/json 头文件已安装: ${install_prefix}"
        if [ -f "${install_prefix}/share/cmake/nlohmann_json/nlohmann_jsonConfig.cmake" ] \
            || [ -f "${install_prefix}/lib/cmake/nlohmann_json/nlohmann_jsonConfig.cmake" ]; then
            log "  CMake 配置已安装"
        fi
        return 0
    fi

    log "⚠ 未找到 nlohmann/json.hpp，请检查: ${install_prefix}"
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

    log "平台: ${PLATFORM}  形态: header-only  版本: ${NLOHMANN_JSON_VERSION}  线程: ${BUILD_JOBS}"
    log "安装路径: ${INSTALL_DIR}/${PLATFORM}/nlohmann_json"
    log "工具链: ${TOOLCHAIN_FILE}"

    prepare_third_party_git_source_at_ref nlohmann_json \
        "${NLOHMANN_JSON_GIT_URL}" "${NLOHMANN_JSON_VERSION}" nlohmann_json
    configure_build_install
    verify_installation

    local duration=$(( $(date +%s) - start_time ))
    log "nlohmann_json 构建完成 (耗时 ${duration}s)"
}

main "$@"
