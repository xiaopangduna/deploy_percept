#!/bin/bash
# 第三方库构建器：OpenCV
# 可以单独运行，也可以由 third_party_builder.sh 调用
#
# 下载失败时查看: tmp/opencv/build_<platform>/CMakeDownloadLog.txt

set -e

readonly OPENCV_GIT_URL="https://gitee.com/opencv/opencv.git"
# 与 third_party/armv7l-SSC375/prebuild_libs/opencv 中 libopencv_*.a 对齐
readonly SSC375_OPENCV_MODULES="core,flann,imgproc,ml,photo,dnn,features2d,imgcodecs,videoio,calib3d,highgui,objdetect,stitching,video"

PLATFORM=""
PROJECT_ROOT=""
INSTALL_DIR=""
TOOLCHAIN_FILE=""
TOOLCHAIN_ROOT=""
TOOLCHAIN_PREFIX=""
BUILD_JOBS=""
USE_TOOLCHAIN_CC=1
OPENCV_VERSION="4.5.4"
CROSS_COMPILE_PREFIX=""

log() { echo "[OpenCV构建器] $*"; }

show_help() {
    cat <<EOF
OpenCV 构建器脚本

用法:
  bash $0 [选项]

必需选项:
  --platform <平台>          目标平台 (aarch64, x86_64, armv7l-SSC375)

可选选项:
  --project-root <路径>      项目根目录 (默认: 当前目录)
  --install-dir <路径>       安装目录 (默认: \$PROJECT_ROOT/third_party)
  --toolchain-file <文件>    CMake 工具链 (默认: \$PROJECT_ROOT/cmake/\$PLATFORM-toolchain.cmake)
  --jobs <N>                 并行编译线程数 (默认: 平台相关)
  --help                     显示此帮助信息

OpenCV 版本:
  aarch64 / x86_64           4.5.4
  armv7l-SSC375              4.1.1 (与 SDK 预编译一致)

示例:
  bash scripts/third_party_builders/builder_opencv.sh --platform x86_64
  bash scripts/third_party_builders/builder_opencv.sh --platform armv7l-SSC375 --jobs 4
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
            OPENCV_VERSION="4.5.4"
            ;;
        x86_64)
            CROSS_COMPILE_PREFIX="x86_64-linux-gnu"
            OPENCV_VERSION="4.5.4"
            ;;
        armv7l-SSC375)
            OPENCV_VERSION="4.1.1"
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
        TOOLCHAIN_FILE="${PROJECT_ROOT}/cmake/${PLATFORM}-toolchain.cmake"
    fi

    if [ ! -f "${TOOLCHAIN_FILE}" ]; then
        log "警告: 工具链文件不存在: ${TOOLCHAIN_FILE}"
    fi
}

# 从 cmake/<platform>-toolchain.cmake 解析 TOOLCHAIN_ROOT / TOOLCHAIN_PREFIX（SSC375 唯一来源）
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
    git config --global --add safe.directory "${PROJECT_ROOT}/tmp/opencv" 2>/dev/null || true
    local git_dir
    for git_dir in "${PROJECT_ROOT}/tmp/"*/; do
        if [ -d "${git_dir}.git" ]; then
            git config --global --add safe.directory "${git_dir}" 2>/dev/null || true
        fi
    done
}

prepare_opencv_source() {
    mkdir -p "${PROJECT_ROOT}/tmp" "${PROJECT_ROOT}/third_party"
    setup_git_safe_directories

    cd "${PROJECT_ROOT}/tmp"
    if [ ! -d "opencv" ]; then
        log "克隆 OpenCV 源码..."
        git clone "${OPENCV_GIT_URL}"
    else
        log "OpenCV 目录已存在，跳过克隆"
    fi

    cd opencv
    log "切换到版本 ${OPENCV_VERSION}..."
    git checkout "${OPENCV_VERSION}"
}

build_opencv_cmake_args() {
    OPENCV_CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/${PLATFORM}/opencv"
        -DOPENCV_DOWNLOAD_PATH="${OPENCV_CACHE_DIR}"
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}"
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_PNG=ON
        -DPNG_LIBRARY=""
        -DPNG_PNG_INCLUDE_DIR=""
    )

    if [ "${PLATFORM}" = "armv7l-SSC375" ]; then
        OPENCV_CMAKE_ARGS+=(
            -DBUILD_TESTS=OFF
            -DBUILD_PERF_TESTS=OFF
            -DBUILD_EXAMPLES=OFF
            -DBUILD_opencv_apps=OFF
            -DBUILD_LIST="${SSC375_OPENCV_MODULES}"
            -DWITH_GTK=OFF
            -DWITH_QT=OFF
            -DWITH_FFMPEG=OFF
            -DWITH_GSTREAMER=OFF
            -DWITH_OPENGL=OFF
            -DWITH_OPENCL=OFF
            -DWITH_CUDA=OFF
            -DWITH_V4L=ON
            -DCPU_BASELINE=NEON
            -DCPU_DISPATCH=
            -DCPU_BASELINE_DISABLE=VFPV3
        )
    fi
}

configure_build_install() {
    local build_dir="build_${PLATFORM}"
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    cd "${build_dir}"

    OPENCV_CACHE_DIR="../../opencv_${PLATFORM}_cache"
    mkdir -p "${OPENCV_CACHE_DIR}"

    build_opencv_cmake_args
    log "配置 CMake (缓存: ${OPENCV_CACHE_DIR})..."
    cmake .. "${OPENCV_CMAKE_ARGS[@]}"

    log "编译 OpenCV (-j${BUILD_JOBS})..."
    make -j"${BUILD_JOBS}"

    log "安装 OpenCV..."
    make install
}

verify_installation() {
    local install_prefix="${INSTALL_DIR}/${PLATFORM}/opencv"
    local lib_dir=""

    if [ -f "${install_prefix}/lib/libopencv_core.a" ] || \
       [ -f "${install_prefix}/lib/libopencv_core.so" ]; then
        lib_dir="${install_prefix}/lib"
    elif [ -f "${install_prefix}/lib64/libopencv_core.a" ] || \
         [ -f "${install_prefix}/lib64/libopencv_core.so" ]; then
        lib_dir="${install_prefix}/lib64"
    else
        log "⚠ 未找到 libopencv_core，请检查: ${install_prefix}"
        return 1
    fi

    log "✓ OpenCV 已安装到 ${install_prefix}"

    if [ "${PLATFORM}" = "armv7l-SSC375" ]; then
        local mod missing=0
        IFS=',' read -ra modules <<< "${SSC375_OPENCV_MODULES}"
        for mod in "${modules[@]}"; do
            if [ ! -f "${lib_dir}/libopencv_${mod}.a" ]; then
                log "✗ 缺少模块库: libopencv_${mod}.a"
                missing=1
            fi
        done
        if [ "${missing}" -eq 0 ]; then
            log "✓ 14 个模块库与 SDK 预编译列表一致"
        else
            return 1
        fi
    fi
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

    log "平台: ${PLATFORM}  版本: ${OPENCV_VERSION}  线程: ${BUILD_JOBS}"
    log "安装路径: ${INSTALL_DIR}/${PLATFORM}/opencv"
    log "工具链: ${TOOLCHAIN_FILE}"

    prepare_opencv_source
    configure_build_install
    verify_installation

    local duration=$(( $(date +%s) - start_time ))
    log "OpenCV 构建完成 (耗时 ${duration}s)"
}

main "$@"
