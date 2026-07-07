#!/bin/bash
# 第三方库构建器公共函数

# 与 cmake/toolchains/<platform>-toolchain.cmake 同名
THIRD_PARTY_SUPPORTED_PLATFORMS=(
    x86_64
    aarch64-linux-gnu_orange_pi_4_pro_a733
)

# 各 builder 定义 log() 后可选：builder_log() { log "$@"; }
_builder_log() {
    if declare -F builder_log &>/dev/null; then
        builder_log "$@"
    elif declare -F log &>/dev/null; then
        log "$@"
    else
        echo "$@"
    fi
}

list_supported_third_party_platforms() {
    local platform
    for platform in "${THIRD_PARTY_SUPPORTED_PLATFORMS[@]}"; do
        echo "  ${platform}"
    done
}

validate_third_party_platform() {
    local platform="$1"
    case "${platform}" in
        x86_64|aarch64-linux-gnu_orange_pi_4_pro_a733)
            return 0
            ;;
        *)
            echo "错误: 不支持的平台 '${platform}'"
            echo "可用平台（cmake/toolchains/ 下名称）:"
            list_supported_third_party_platforms
            exit 1
            ;;
    esac
}

# 设置 USE_TOOLCHAIN_CC、CROSS_COMPILE_PREFIX、BUILD_JOBS（需已设 PLATFORM）
configure_third_party_platform() {
    validate_third_party_platform "${PLATFORM}"
    USE_TOOLCHAIN_CC=0
    CROSS_COMPILE_PREFIX=""

    if [ -z "${BUILD_JOBS}" ]; then
        BUILD_JOBS=$(default_third_party_build_jobs)
    fi
}

# 解析 builder 通用参数；调用方须定义 show_help。
# THIRD_PARTY_IGNORE_LINK_TYPE=1 时忽略 --link-type（cnpy/zlib）。
parse_third_party_builder_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build-mode)     shift 2 ;;
            --platform)       PLATFORM="$2"; shift 2 ;;
            --project-root)   PROJECT_ROOT="$2"; shift 2 ;;
            --install-dir)    INSTALL_DIR="$2"; shift 2 ;;
            --toolchain-file) TOOLCHAIN_FILE="$2"; shift 2 ;;
            --link-type)
                if [[ "${THIRD_PARTY_IGNORE_LINK_TYPE:-0}" == 1 ]]; then
                    shift 2
                else
                    LINK_TYPE="$2"
                    shift 2
                fi
                ;;
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

require_third_party_platform_arg() {
    if [ -z "${PLATFORM}" ]; then
        echo "错误: 必须指定平台 (--platform)"
        show_help
        exit 1
    fi
}

# 需已设 PLATFORM；初始化 PROJECT_ROOT/INSTALL_DIR/TOOLCHAIN_FILE/TMP_MODULES_DIR
setup_third_party_paths() {
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
        _builder_log "警告: 工具链文件不存在: ${TOOLCHAIN_FILE}"
    fi

    init_modules_tmp
}

# 从 TOOLCHAIN_FILE 解析 TOOLCHAIN_ROOT、TOOLCHAIN_PREFIX
read_toolchain_vars() {
    local key value line
    if [ ! -f "${TOOLCHAIN_FILE}" ]; then
        _builder_log "错误: 工具链文件不存在: ${TOOLCHAIN_FILE}"
        return 1
    fi

    for key in TOOLCHAIN_ROOT TOOLCHAIN_PREFIX; do
        line=$(grep -E "^set\\(${key} " "${TOOLCHAIN_FILE}" | head -1)
        if [ -z "${line}" ]; then
            _builder_log "错误: ${TOOLCHAIN_FILE} 中未找到 set(${key} ...)"
            return 1
        fi
        value=$(sed -E 's/^set\([^ ]+ "([^"]+)".*/\1/' <<< "${line}")
        if [ -z "${value}" ] || [ "${value}" = "${line}" ]; then
            _builder_log "错误: 无法解析 ${key}（${TOOLCHAIN_FILE}）"
            return 1
        fi
        printf -v "${key}" '%s' "${value}"
    done
}

# mode: cmake（默认，由 CMAKE_TOOLCHAIN_FILE 驱动）| configure（configure/make，导出 CC/AR）
setup_third_party_toolchain_env() {
    local mode="${1:-cmake}"
    local compiler=""

    if [[ "${PLATFORM}" == aarch64-linux-gnu_orange_pi_4_pro_a733 ]]; then
        read_toolchain_vars || exit 1
        CROSS_COMPILE_PREFIX="${TOOLCHAIN_PREFIX}"
        export PATH="${TOOLCHAIN_ROOT}/bin:${PATH}"
    fi

    case "${mode}" in
        configure)
            if [ -n "${CROSS_COMPILE_PREFIX}" ]; then
                export CC="${CROSS_COMPILE_PREFIX}-gcc"
                export CXX="${CROSS_COMPILE_PREFIX}-g++"
                export AR="${CROSS_COMPILE_PREFIX}-ar"
                export RANLIB="${CROSS_COMPILE_PREFIX}-ranlib"
            fi
            compiler="${CC:-gcc}"
            if [ -n "${CROSS_COMPILE_PREFIX}" ] && ! command -v "${compiler}" &>/dev/null; then
                _builder_log "警告: 未找到 ${compiler}"
                _builder_log "  请挂载工具链: ${TOOLCHAIN_ROOT}"
            fi
            ;;
        cmake|*)
            if [ "${USE_TOOLCHAIN_CC}" = "1" ] && [ -n "${CROSS_COMPILE_PREFIX}" ]; then
                export CC="${CROSS_COMPILE_PREFIX}-gcc"
                export CXX="${CROSS_COMPILE_PREFIX}-g++"
            fi
            if [ -n "${CROSS_COMPILE_PREFIX}" ] && ! command -v "${CROSS_COMPILE_PREFIX}-g++" &>/dev/null; then
                _builder_log "警告: 未找到 ${CROSS_COMPILE_PREFIX}-g++"
                _builder_log "  请挂载工具链: ${TOOLCHAIN_ROOT}"
            fi
            ;;
    esac
}

prepare_third_party_git_source() {
    local repo_dir="$1"
    local git_url="$2"
    local display_name="${3:-${repo_dir}}"

    setup_git_safe_directories
    cd "${TMP_MODULES_DIR}"
    if [ ! -d "${repo_dir}" ]; then
        _builder_log "克隆 ${display_name} 源码..."
        git clone "${git_url}" "${repo_dir}"
    else
        _builder_log "${repo_dir} 目录已存在，跳过克隆"
    fi
}

# 克隆（或复用）源码并 checkout 到指定 ref（tag/branch/commit）
prepare_third_party_git_source_at_ref() {
    local repo_dir="$1"
    local git_url="$2"
    local git_ref="$3"
    local display_name="${4:-${repo_dir}}"

    prepare_third_party_git_source "${repo_dir}" "${git_url}" "${display_name}"
    cd "${TMP_MODULES_DIR}/${repo_dir}"
    _builder_log "切换到版本 ${git_ref}..."
    git checkout "${git_ref}"
}

prepare_third_party_build_dir() {
    local install_prefix="$1"
    local build_dir="$2"

    clean_third_party_install_prefix "${install_prefix}"
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    cd "${build_dir}"
}

# 需先设置 PROJECT_ROOT
init_modules_tmp() {
    TMP_MODULES_DIR="${PROJECT_ROOT}/tmp/modules"
    mkdir -p "${TMP_MODULES_DIR}" "${PROJECT_ROOT}/third_party"
}

setup_git_safe_directories() {
    git config --global --add safe.directory "${PROJECT_ROOT}/tmp" 2>/dev/null || true
    git config --global --add safe.directory "${TMP_MODULES_DIR}" 2>/dev/null || true
    local git_dir
    for git_dir in "${TMP_MODULES_DIR}/"*/; do
        if [ -d "${git_dir}.git" ]; then
            git config --global --add safe.directory "${git_dir}" 2>/dev/null || true
        fi
    done
}

link_type_log_warn() {
    echo "[third_party] 警告: $*" >&2
}

# 默认并行编译线程：min(nproc, 8)；nproc 不可用时为 8
default_third_party_build_jobs() {
    local nproc_val
    nproc_val=$(nproc 2>/dev/null || echo 8)
    if [ "${nproc_val}" -gt 8 ]; then
        echo 8
    else
        echo "${nproc_val}"
    fi
}

# 校验链接类型；默认 static。可传入空值以使用 THIRD_PARTY_LINK_TYPE 环境变量。
validate_link_type() {
    local requested="${1:-${THIRD_PARTY_LINK_TYPE:-static}}"
    case "${requested}" in
        static|shared)
            echo "${requested}"
            ;;
        *)
            echo "错误: 链接类型必须是 static 或 shared: ${requested}" >&2
            exit 1
            ;;
    esac
}

# 构建前清除该库 install prefix，避免 static/shared 产物并存。
clean_third_party_install_prefix() {
    local prefix="$1"
    if [[ -d "${prefix}" ]]; then
        echo "[third_party] 清除旧安装: ${prefix}"
        rm -rf "${prefix}"
    fi
}

# 预编译库仅 shared；请求 static 时降级并警告。
resolve_effective_link_type() {
    local lib="$1"
    local requested="$2"

    case "${lib}" in
        rga|rknpu)
            if [[ "${requested}" == static ]]; then
                link_type_log_warn "${lib} 不支持 static，将安装 shared 预编译库（忽略 --link static）"
            fi
            echo shared
            ;;
        *)
            echo "${requested}"
            ;;
    esac
}

cmake_bool_from_link_type() {
    local link_type="$1"
    if [[ "${link_type}" == shared ]]; then
        echo ON
    else
        echo OFF
    fi
}

# 按 link_type（static|shared）在 third_party zlib prefix 中查找 libz。
find_zlib_library_path() {
    local zlib_root="$1"
    local link_type="${2:-static}"
    local candidate

    if [[ "${link_type}" == shared ]]; then
        for candidate in \
            "${zlib_root}/lib/libz.so" \
            "${zlib_root}/lib64/libz.so"
        do
            if [[ -f "${candidate}" ]]; then
                echo "${candidate}"
                return 0
            fi
        done
    else
        for candidate in \
            "${zlib_root}/lib/libz.a" \
            "${zlib_root}/lib64/libz.a"
        do
            if [[ -f "${candidate}" ]]; then
                echo "${candidate}"
                return 0
            fi
        done
    fi

    return 1
}

# prefix link_type lib_stem（如 libz、libcnpy、libopencv_core）
verify_installed_library() {
    local prefix="$1"
    local link_type="$2"
    local lib_stem="$3"
    local lib_dir="" name=""

    for lib_dir in "${prefix}/lib" "${prefix}/lib64"; do
        [[ -d "${lib_dir}" ]] || continue
        if [[ "${link_type}" == shared ]]; then
            if [[ -f "${lib_dir}/${lib_stem}.so" ]]; then
                return 0
            fi
            for name in "${lib_dir}/${lib_stem}.so."*; do
                if [[ -f "${name}" ]]; then
                    return 0
                fi
            done
        else
            if [[ -f "${lib_dir}/${lib_stem}.a" ]]; then
                return 0
            fi
        fi
    done

    return 1
}

# 部分构建系统会同时安装 .a 与 .so；按 link_type 删除多余产物（zlib/cnpy 不使用）。
scrub_opposite_library_artifacts() {
    local prefix="$1"
    local link_type="$2"
    local lib_stem="$3"
    local lib_dir

    for lib_dir in "${prefix}/lib" "${prefix}/lib64"; do
        [[ -d "${lib_dir}" ]] || continue
        if [[ "${link_type}" == shared ]]; then
            rm -f "${lib_dir}/${lib_stem}.a"
        else
            rm -f "${lib_dir}/${lib_stem}.so"
            rm -f "${lib_dir}/${lib_stem}.so."*
        fi
    done
}

# OpenCV install 可能把 ZLIB 绝对路径写入 OpenCVModules.cmake；不修改官方产物，仅警告。
warn_opencv_zlib_absolute_paths() {
    local modules_file="$1"

    if [[ ! -f "${modules_file}" ]]; then
        return 0
    fi

    if grep -qE '(^|[;"[:space:]])/[^";[:space:]]+/zlib/lib/libz\.(a|so)' "${modules_file}"; then
        _builder_log "警告: OpenCVModules.cmake 中含有 zlib 绝对路径"
        _builder_log "  工程目录迁移后 find_package(OpenCV) 可能链接失败，请在目标路径重新构建 OpenCV"
        _builder_log "  文件: ${modules_file}"
    fi
}
