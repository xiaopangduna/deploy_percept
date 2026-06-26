#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET="x86_64-debug"
INSTALL_DIR=""
BOARD=""

PERCEPT_TESTS_REL="share/percept/tests"

PERCEPT_INSTALL_TESTS=(
    smoke_tests
    test_YoloV5DetectPostProcess
    test_YoloV5SegPostProcess
    test_YoloV8SegPostProcess
    test_yolov5_detect_awnn
)

show_help() {
    echo "deploy_percept 测试脚本（仅测试，不构建、不安装）"
    echo ""
    echo "用法: bash $0 [选项]"
    echo ""
    echo "模式 A — build tree（ctest）:"
    echo "  bash $0 [--preset <name>]"
    echo ""
    echo "模式 B — install tree（本地 prefix）:"
    echo "  bash $0 --install-dir <path>"
    echo ""
    echo "模式 C — 开发板（install 已 rsync 到板子）:"
    echo "  bash $0 --board <user@host:remote_path>"
    echo ""
    echo "选项:"
    echo "  --preset <name>            CMake preset（默认: x86_64-debug，用于 ctest）"
    echo "  --install-dir <path>       对 install prefix 跑 share/percept/tests/ 内测试"
    echo "  --board <user@host:path>   SSH 到开发板跑 install tree 测试"
    echo "  --help                     显示帮助"
    echo ""
    echo "示例:"
    echo "  bash scripts/build.sh --preset x86_64-debug"
    echo "  bash $0 --preset x86_64-debug"
    echo "  bash $0 --install-dir install/x86_64"
    echo "  bash $0 --board orangepi@192.168.1.10:~/deploy_percept"
    echo "  bash $0 --board orangepi@192.168.1.10:/home/orangepi/deploy_percept"
}

run_install_tree_tests() {
    local install_dir="$1"
    if [[ ! -d "${install_dir}/share/percept" ]]; then
        echo "错误: 无效的 install 目录（缺少 share/percept/）: ${install_dir}" >&2
        exit 1
    fi
    install_dir="$(cd "${install_dir}" && pwd)"

    local tests_dir="${install_dir}/${PERCEPT_TESTS_REL}"
    if [[ ! -d "${tests_dir}" ]]; then
        echo "错误: 未找到测试目录: ${tests_dir}" >&2
        echo "提示: 构建时需 ENABLE_TESTS=ON 且 INSTALL_TESTS=ON，并执行 cmake --install" >&2
        exit 1
    fi

    export PERCEPT_ROOT="${install_dir}/share/percept"
    export PERCEPT_OUTPUT_DIR="${install_dir}/var/percept/output"
    export LD_LIBRARY_PATH="${install_dir}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    mkdir -p "${PERCEPT_OUTPUT_DIR}"

    echo "=== install tree tests ==="
    echo "prefix:    ${install_dir}"
    echo "tests dir: ${tests_dir}"

    local fail=0
    local t
    for t in "${PERCEPT_INSTALL_TESTS[@]}"; do
        if [[ ! -x "${tests_dir}/${t}" ]]; then
            echo "skip (not installed): ${t}"
            continue
        fi
        echo "=== ${t} ==="
        "${tests_dir}/${t}" || fail=1
    done
    return "${fail}"
}

run_install_tree_tests_remote() {
    local board_spec="$1"
    if [[ ! "${board_spec}" =~ ^([^@]+@[^:]+):(.+)$ ]]; then
        echo "错误: --board 格式应为 user@host:remote_path" >&2
        exit 1
    fi
    local ssh_host="${BASH_REMATCH[1]}"
    local remote_dir_input="${BASH_REMATCH[2]}"

    # ~ 在 SSH 远程双引号字符串中不会展开，先解析为绝对路径
    local remote_dir
    remote_dir="$(ssh "${ssh_host}" "cd ${remote_dir_input} && pwd")" || {
        echo "错误: 无法访问远程目录: ${remote_dir_input}" >&2
        exit 1
    }

    local tests_dir="${remote_dir}/${PERCEPT_TESTS_REL}"

    echo "=== install tree tests (remote) ==="
    echo "host:      ${ssh_host}"
    echo "prefix:    ${remote_dir}"
    echo "tests dir: ${tests_dir}"

    local tests_shell=""
    local t
    for t in "${PERCEPT_INSTALL_TESTS[@]}"; do
        tests_shell+="if [[ -x \"${tests_dir}/${t}\" ]]; then echo '=== ${t} ==='; \"${tests_dir}/${t}\" || FAIL=1; else echo 'skip (not installed): ${t}'; fi;"
    done

    ssh "${ssh_host}" "set -euo pipefail
FAIL=0
export PERCEPT_ROOT=\"${remote_dir}/share/percept\"
export PERCEPT_OUTPUT_DIR=\"${remote_dir}/var/percept/output\"
export LD_LIBRARY_PATH=\"${remote_dir}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"
mkdir -p \"\${PERCEPT_OUTPUT_DIR}\"
if [[ ! -d \"${tests_dir}\" ]]; then
  echo \"错误: 未找到测试目录: ${tests_dir}\" >&2
  echo \"提示: INSTALL_TESTS=ON 且已 cmake --install\" >&2
  exit 1
fi
${tests_shell}
exit \${FAIL}"
}

run_ctest() {
    local build_dir="${PROJECT_ROOT}/build/${PRESET}"
    if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
        echo "错误: 构建目录未配置: ${build_dir}" >&2
        echo "请先执行: bash scripts/build.sh --preset ${PRESET}" >&2
        exit 1
    fi
    echo "=== ctest (build tree) ==="
    echo "preset:    ${PRESET}"
    echo "build dir: ${build_dir}"
    ctest --test-dir "${build_dir}" --output-on-failure
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset) PRESET="$2"; shift 2 ;;
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        --board) BOARD="$2"; shift 2 ;;
        --help) show_help; exit 0 ;;
        *)
            echo "错误: 未知参数: $1"
            echo "提示: 构建请用 scripts/build.sh，安装请用 scripts/install.sh"
            show_help
            exit 1
            ;;
    esac
done

if [[ -n "${BOARD}" && -n "${INSTALL_DIR}" ]]; then
    echo "错误: --board 与 --install-dir 不能同时使用" >&2
    exit 1
fi

if [[ -n "${BOARD}" ]]; then
    run_install_tree_tests_remote "${BOARD}"
    exit $?
fi

if [[ -n "${INSTALL_DIR}" ]]; then
    if [[ "${INSTALL_DIR}" != /* ]]; then
        INSTALL_DIR="${PROJECT_ROOT}/${INSTALL_DIR}"
    fi
    run_install_tree_tests "${INSTALL_DIR}"
    exit $?
fi

run_ctest
