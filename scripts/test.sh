#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET="x86_64-debug"
BOARD=""
RUN_BENCH=0
BENCH_LOOPS=50

SSH_WRAP=()
RSYNC_RSH=""

show_help() {
    echo "deploy_percept 测试脚本（build + install + 验证 install 包）"
    echo ""
    echo "用法: bash $0 --preset <name> [选项]"
    echo ""
    echo "流程:"
    echo "  PC 模式:     build → install → scripts/run_install_tests.sh"
    echo "  开发板模式:  build → install → rsync → SSH 远程 run_install_tests.sh"
    echo ""
    echo "选项:"
    echo "  --preset <name>          CMake preset（默认: x86_64-debug）"
    echo "  --board <user@host:path> 开发板 install 根目录（启用板端模式）"
    echo "  --bench [loops]          传给 run_install_tests.sh（默认 loops=50）"
    echo "  --help                   显示帮助"
    echo ""
    echo "SSH 认证:"
    echo "  推荐: ssh-copy-id 配置密钥，无需额外环境变量"
    echo "  密码: export SSHPASS='...'  （需安装 sshpass；仅当前 shell，勿提交仓库）"
    echo ""
    echo "示例:"
    echo "  bash $0 --preset x86_64-debug"
    echo "  bash $0 --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \\"
    echo "    --board orangepi@192.168.1.10:~/deploy_percept"
    echo "  export SSHPASS='orangepi'"
    echo "  bash $0 --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \\"
    echo "    --board orangepi@192.168.1.10:~/deploy_percept --bench 100"
    echo ""
    echo "install 包内直接跑: ./scripts/run_install_tests.sh [--bench [loops]]"
    echo "开发期 build tree GTest: cd build/<preset> && ctest --output-on-failure"
}

is_unsigned_loops() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

setup_ssh_transport() {
    if [[ -n "${SSHPASS:-}" ]]; then
        if ! command -v sshpass >/dev/null 2>&1; then
            echo "错误: 已设置 SSHPASS 但未找到 sshpass" >&2
            echo "提示: sudo apt install sshpass，或改用 ssh-copy-id 密钥登录" >&2
            exit 1
        fi
        SSH_WRAP=(sshpass -e ssh -o StrictHostKeyChecking=accept-new)
        RSYNC_RSH="sshpass -e ssh -o StrictHostKeyChecking=accept-new"
        echo "SSH: 使用 SSHPASS + sshpass"
    else
        SSH_WRAP=(ssh -o StrictHostKeyChecking=accept-new)
        RSYNC_RSH="ssh -o StrictHostKeyChecking=accept-new"
        echo "SSH: 使用默认 ssh（密钥或 ~/.ssh/config）"
    fi
}

run_build() {
    echo ""
    echo "==============================="
    echo "=== build ==="
    echo "==============================="
    bash "${SCRIPT_DIR}/build.sh" --preset "${PRESET}"
}

run_install() {
    echo ""
    echo "==============================="
    echo "=== install ==="
    echo "==============================="
    bash "${SCRIPT_DIR}/install.sh" --preset "${PRESET}"
}

resolve_install_dir() {
    local build_dir="${PROJECT_ROOT}/build/${PRESET}"
    if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
        echo "错误: 构建目录未配置: ${build_dir}" >&2
        exit 1
    fi
    local prefix
    prefix="$(grep -E '^CMAKE_INSTALL_PREFIX:' "${build_dir}/CMakeCache.txt" | cut -d= -f2)"
    if [[ -z "${prefix}" || ! -d "${prefix}" ]]; then
        echo "错误: install prefix 不存在: ${prefix}" >&2
        exit 1
    fi
    echo "${prefix}"
}

runner_args() {
    RUNNER_ARGS=()
    if [[ "${RUN_BENCH}" -eq 1 ]]; then
        RUNNER_ARGS=(--bench "${BENCH_LOOPS}")
    fi
}

parse_board_spec() {
    local board_spec="$1"
    if [[ ! "${board_spec}" =~ ^([^@]+@[^:]+):(.+)$ ]]; then
        echo "错误: --board 格式应为 user@host:remote_path" >&2
        exit 1
    fi
    SSH_HOST="${BASH_REMATCH[1]}"
    REMOTE_DIR_INPUT="${BASH_REMATCH[2]}"
}

run_install_tests_local() {
    local install_dir="$1"
    local runner="${install_dir}/scripts/run_install_tests.sh"
    if [[ ! -x "${runner}" ]]; then
        echo "错误: 未找到 runner: ${runner}" >&2
        echo "提示: INSTALL_TESTS=ON 且已 install" >&2
        exit 1
    fi
    runner_args
    "${runner}" "${RUNNER_ARGS[@]}"
}

run_rsync_to_board() {
    local install_dir="$1"
    local ssh_host="$2"
    local remote_dir_input="$3"

    echo ""
    echo "=== rsync install → board ==="
    echo "source:  ${install_dir}/"
    echo "target:  ${ssh_host}:${remote_dir_input}/"

    rsync -avz --delete -e "${RSYNC_RSH}" "${install_dir}/" "${ssh_host}:${remote_dir_input}/"
}

run_install_tests_remote() {
    local ssh_host="$1"
    local remote_dir_input="$2"

    local remote_dir
    remote_dir="$("${SSH_WRAP[@]}" "${ssh_host}" "cd ${remote_dir_input} && pwd")" || {
        echo "错误: 无法访问远程目录: ${remote_dir_input}" >&2
        exit 1
    }

    runner_args

    echo ""
    echo "=== run_install_tests.sh (remote) ==="
    echo "host: ${ssh_host}"
    echo "dir:  ${remote_dir_input} → ${remote_dir}"

    local remote_runner="${remote_dir}/scripts/run_install_tests.sh"
    local remote_cmd
    remote_cmd="$(printf '%q' "${remote_runner}")"
    local arg
    for arg in "${RUNNER_ARGS[@]}"; do
        remote_cmd+=" $(printf '%q' "${arg}")"
    done

    "${SSH_WRAP[@]}" "${ssh_host}" "bash -lc $(printf '%q' "${remote_cmd}")"
}

run_board_suite() {
    local board_spec="$1"
    local install_dir="$2"

    parse_board_spec "${board_spec}"
    setup_ssh_transport
    run_rsync_to_board "${install_dir}" "${SSH_HOST}" "${REMOTE_DIR_INPUT}"
    run_install_tests_remote "${SSH_HOST}" "${REMOTE_DIR_INPUT}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset) PRESET="$2"; shift 2 ;;
        --board) BOARD="$2"; shift 2 ;;
        --bench)
            RUN_BENCH=1
            shift
            if [[ $# -gt 0 ]] && is_unsigned_loops "$1"; then
                BENCH_LOOPS="$1"
                shift
            fi
            ;;
        --help) show_help; exit 0 ;;
        *)
            echo "错误: 未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "==============================="
echo "deploy_percept test"
echo "  preset: ${PRESET}"
if [[ -n "${BOARD}" ]]; then
    echo "  mode:   board (${BOARD})"
else
    echo "  mode:   PC (local install)"
fi
if [[ "${RUN_BENCH}" -eq 1 ]]; then
    echo "  bench:  ${BENCH_LOOPS} loops"
fi
echo "==============================="

run_build
run_install

INSTALL_DIR="$(resolve_install_dir)"

if [[ -n "${BOARD}" ]]; then
    run_board_suite "${BOARD}" "${INSTALL_DIR}"
    exit $?
fi

run_install_tests_local "${INSTALL_DIR}"
exit $?
