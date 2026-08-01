#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET="x86_64-debug"
BOARD=""
RUN_BENCH=0
BENCH_LOOPS=50
RUN_PROFILE=0
PROFILE_DURATION_SEC=60

SSH_WRAP=()
RSYNC_RSH=""

show_help() {
    echo "deploy_percept 测试脚本（编排：build+install → test_runner）"
    echo ""
    echo "用法: bash $0 [选项]"
    echo ""
    echo "默认 host: 本地 install + test_runner"
    echo "--board:   rsync 后远程 test_runner"
    echo ""
    echo "选项:"
    echo "  --preset <name>          CMake preset（默认: x86_64-debug）"
    echo "  --board <user@host:path> 板端模式（user@host:远程 install 根目录）"
    echo "  --bench [loops]          传给 test_runner（默认 loops=50）"
    echo "  --profile [sec]          传给 test_runner：app 循环墙钟秒数（默认 60）"
    echo "  --help                   显示帮助"
    echo ""
    echo "板端 SSH: 仅支持密码登录"
    echo "  export SSHPASS='...'   # 需安装 sshpass；仅当前 shell，勿提交仓库"
    echo ""
    echo "示例:"
    echo "  bash $0 --preset x86_64-debug"
    echo "  export SSHPASS='orangepi'"
    echo "  bash $0 --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \\"
    echo "    --board root@192.168.1.10:~/deploy_percept --bench 100 --profile 60"
}

is_unsigned() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

setup_ssh_transport() {
    if [[ -z "${SSHPASS:-}" ]]; then
        echo "错误: 板端模式需要设置 SSHPASS" >&2
        echo "示例: export SSHPASS='your_password'" >&2
        exit 1
    fi
    if ! command -v sshpass >/dev/null 2>&1; then
        echo "错误: 未找到 sshpass，请先安装（如: sudo apt install sshpass）" >&2
        exit 1
    fi
    SSH_WRAP=(sshpass -e ssh -o StrictHostKeyChecking=accept-new)
    RSYNC_RSH="sshpass -e ssh -o StrictHostKeyChecking=accept-new"
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

build_runner_args() {
    RUNNER_ARGS=()
    if [[ "${RUN_BENCH}" -eq 1 ]]; then
        RUNNER_ARGS+=(--bench "${BENCH_LOOPS}")
    fi
    if [[ "${RUN_PROFILE}" -eq 1 ]]; then
        RUNNER_ARGS+=(--profile "${PROFILE_DURATION_SEC}")
    fi
}

run_test_runner_local() {
    local install_dir="$1"
    local runner="${install_dir}/scripts/test_runner.sh"
    if [[ ! -x "${runner}" ]]; then
        echo "错误: 未找到 runner: ${runner}" >&2
        echo "提示: INSTALL_TESTS=ON 且已 install" >&2
        exit 1
    fi
    build_runner_args
    "${runner}" "${RUNNER_ARGS[@]}"
}

run_on_board() {
    local install_dir="$1"
    local board_spec="$2"

    if [[ ! "${board_spec}" =~ ^([^@]+@[^:]+):(.+)$ ]]; then
        echo "错误: --board 格式应为 user@host:remote_path" >&2
        exit 1
    fi
    local ssh_host="${BASH_REMATCH[1]}"
    local remote_dir="${BASH_REMATCH[2]}"

    setup_ssh_transport
    build_runner_args

    local artifacts_dir="${PROJECT_ROOT}/ci-artifacts/board-logs"
    mkdir -p "${artifacts_dir}"
    rm -rf "${artifacts_dir:?}/"*

    echo "=== rsync → ${ssh_host}:${remote_dir}/ ==="
    rsync -avzc --delete -e "${RSYNC_RSH}" "${install_dir}/" "${ssh_host}:${remote_dir}/"

    # 展开 ~ 等，得到远程绝对路径后再拼 runner
    local remote_abs
    remote_abs="$("${SSH_WRAP[@]}" "${ssh_host}" "cd ${remote_dir} && pwd")" || {
        echo "错误: 无法访问远程目录: ${remote_dir}" >&2
        exit 1
    }

    local remote_cmd
    remote_cmd="$(printf '%q' "${remote_abs}/scripts/test_runner.sh")"
    local arg
    for arg in "${RUNNER_ARGS[@]}"; do
        remote_cmd+=" $(printf '%q' "${arg}")"
    done

    echo "=== test_runner (remote) ==="
    local remote_rc=0
    set +e
    "${SSH_WRAP[@]}" "${ssh_host}" "bash -lc $(printf '%q' "${remote_cmd}")" \
        2>&1 | tee "${artifacts_dir}/test_runner_console.log"
    remote_rc=${PIPESTATUS[0]}
    set -e

    echo "=== pull board logs → ${artifacts_dir}/ ==="
    mkdir -p "${artifacts_dir}/logs"
    if rsync -avz -e "${RSYNC_RSH}" \
        "${ssh_host}:${remote_abs}/var/percept/logs/" \
        "${artifacts_dir}/logs/"; then
        echo "pulled: ${remote_abs}/var/percept/logs/"
    else
        echo "警告: 未能拉取板端 var/percept/logs/" >&2
    fi

    {
        echo "board_spec=${board_spec}"
        echo "remote_abs=${remote_abs}"
        echo "remote_rc=${remote_rc}"
        echo "pulled_at=$(date -Iseconds)"
    } >"${artifacts_dir}/pull_meta.txt"

    echo "artifacts: ${artifacts_dir}"
    return "${remote_rc}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset) PRESET="$2"; shift 2 ;;
        --board) BOARD="$2"; shift 2 ;;
        --bench)
            RUN_BENCH=1
            shift
            if [[ $# -gt 0 ]] && is_unsigned "$1"; then
                BENCH_LOOPS="$1"
                shift
            fi
            ;;
        --profile)
            RUN_PROFILE=1
            shift
            if [[ $# -gt 0 ]] && is_unsigned "$1"; then
                PROFILE_DURATION_SEC="$1"
                shift
            fi
            if [[ "${PROFILE_DURATION_SEC}" -eq 0 ]]; then
                echo "错误: --profile 时长须 > 0" >&2
                exit 1
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

if [[ -n "${BOARD}" ]]; then
    echo "test: preset=${PRESET} mode=board board=${BOARD}"
else
    echo "test: preset=${PRESET} mode=host"
fi
if [[ "${RUN_BENCH}" -eq 1 ]]; then
    echo "test: bench=${BENCH_LOOPS}"
fi
if [[ "${RUN_PROFILE}" -eq 1 ]]; then
    echo "test: profile=${PROFILE_DURATION_SEC}"
fi

bash "${SCRIPT_DIR}/build.sh" --preset "${PRESET}" --install

INSTALL_DIR="$(resolve_install_dir)"

if [[ -n "${BOARD}" ]]; then
    run_on_board "${INSTALL_DIR}" "${BOARD}"
else
    run_test_runner_local "${INSTALL_DIR}"
fi
