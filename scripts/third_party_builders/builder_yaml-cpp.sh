#!/bin/bash
# 第三方库构建器：yaml-cpp
# 可以单独运行，也可以由 third_party_builder.sh 调用

set -e

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

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
LINK_TYPE=""
EFFECTIVE_LINK_TYPE=""

log() { echo "[yaml-cpp构建器] $*"; }

show_help() {
    cat <<EOF
yaml-cpp 构建器脚本

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

yaml-cpp 版本: ${YAML_CPP_VERSION}

示例:
  bash scripts/third_party_builders/builder_yaml-cpp.sh --platform x86_64
  bash scripts/third_party_builders/builder_yaml-cpp.sh --platform aarch64-linux-gnu_orange_pi_4_pro_a733
EOF
}

configure_build_install() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/yaml-cpp"
    local build_dir="${TMP_MODULES_DIR}/yaml-cpp/build_${PLATFORM}"
    local shared_flag

    prepare_third_party_build_dir "${install_prefix}" "${build_dir}"

    shared_flag=$(cmake_bool_from_link_type "${EFFECTIVE_LINK_TYPE}")
    log "配置 CMake (${EFFECTIVE_LINK_TYPE})..."
    cmake "${TMP_MODULES_DIR}/yaml-cpp" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DYAML_CPP_BUILD_TESTS=OFF \
        -DYAML_CPP_BUILD_TOOLS=OFF \
        -DYAML_BUILD_SHARED_LIBS="${shared_flag}"

    log "编译 yaml-cpp (-j${BUILD_JOBS})..."
    cmake --build . -j"${BUILD_JOBS}"

    log "安装 yaml-cpp..."
    cmake --install .
}

verify_installation() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/yaml-cpp"

    if verify_installed_library "${install_prefix}" "${EFFECTIVE_LINK_TYPE}" libyaml-cpp; then
        log "✓ yaml-cpp (${EFFECTIVE_LINK_TYPE}) 已安装到 ${install_prefix}"
        if [ -f "${install_prefix}/lib/cmake/yaml-cpp/yaml-cpp-config.cmake" ]; then
            log "  CMake 配置: ${install_prefix}/lib/cmake/yaml-cpp/yaml-cpp-config.cmake"
        fi
        return 0
    fi

    if [ -f "${install_prefix}/include/yaml-cpp/yaml.h" ]; then
        log "✓ yaml-cpp 头文件已安装: ${install_prefix}"
        return 0
    fi

    log "⚠ 未找到 libyaml-cpp (${EFFECTIVE_LINK_TYPE}) 或头文件，请检查: ${install_prefix}"
    return 1
}

main() {
    local start_time
    start_time=$(date +%s)

    parse_third_party_builder_args "$@"
    require_third_party_platform_arg

    LINK_TYPE=$(validate_link_type "${LINK_TYPE}")
    EFFECTIVE_LINK_TYPE=$(resolve_effective_link_type yaml-cpp "${LINK_TYPE}")

    configure_third_party_platform
    setup_third_party_paths
    setup_third_party_toolchain_env cmake

    log "平台: ${PLATFORM}  链接: ${EFFECTIVE_LINK_TYPE}  版本: ${YAML_CPP_VERSION}  线程: ${BUILD_JOBS}"
    log "安装路径: ${INSTALL_DIR}/${PLATFORM}/yaml-cpp"
    log "工具链: ${TOOLCHAIN_FILE}"

    prepare_third_party_git_source_at_ref yaml-cpp "${YAML_CPP_GIT_URL}" "${YAML_CPP_VERSION}" yaml-cpp
    configure_build_install
    verify_installation

    local duration=$(( $(date +%s) - start_time ))
    log "yaml-cpp 构建完成 (耗时 ${duration}s)"
}

main "$@"
