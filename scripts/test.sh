#!/bin/bash
# 主机编排：build+install →（可选 sync）→ 调 test_run.sh 跑 tests/（可选 benchmarks/）
# 单目录执行见 scripts/test_run.sh；同步见 scripts/sync_install_to_board.sh。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET="x86_64-debug"
BOARD=""
DEPLOY_FULL=0
RUN_BENCH=0

SSH_WRAP=()
SSH_HOST=""
REMOTE_ABS=""

show_help() {
    echo "deploy_percept 测试编排（build+install → 可选同步 → test_run）"
    echo ""
    echo "用法: bash $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --preset <name>   CMake preset（默认: x86_64-debug）"
    echo "  --board <user@host:/abs/.../deploy_percept>  板端（绝对路径，basename=deploy_percept）"
    echo "  --full            全量同步（透传 sync_install_to_board.sh；默认 quick）"
    echo "  --bench           额外跑 install/.../benchmarks/（同 test_run，ELF 默认参数）"
    echo "  --help            显示帮助"
    echo ""
    echo "tests/、benchmarks/ 均由 scripts/test_run.sh 执行（无额外 ELF 参数）。"
    echo "profile 请另行执行（不经本脚本）。"
    echo ""
    echo "板端需: export SSHPASS='...'（sshpass）"
    echo "新板 / CI 请加 --full。--bench 需构建时 ENABLE_BENCHMARKS=ON。"
    echo ""
    echo "示例:"
    echo "  bash $0 --preset x86_64-debug"
    echo "  bash $0 --preset armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release \\"
    echo "    --board root@192.168.1.102:/customer/tmp/deploy_percept --full --bench"
}

setup_ssh_transport() {
    if [[ -z "${SSHPASS:-}" ]]; then
        echo "错误: 板端模式需要设置 SSHPASS" >&2
        return 1
    fi
    if ! command -v sshpass >/dev/null 2>&1; then
        echo "错误: 未找到 sshpass" >&2
        return 1
    fi
    SSH_WRAP=(sshpass -e ssh -n -o StrictHostKeyChecking=accept-new)
}

parse_board_spec() {
    local board_spec="$1"
    if [[ ! "${board_spec}" =~ ^([^@]+@[^:]+):(.+)$ ]]; then
        echo "错误: --board 格式应为 user@host:/absolute/path/deploy_percept" >&2
        return 1
    fi
    SSH_HOST="${BASH_REMATCH[1]}"
    REMOTE_ABS="${BASH_REMATCH[2]}"
    REMOTE_ABS="${REMOTE_ABS%/}"
    if [[ "${REMOTE_ABS}" != /* ]]; then
        echo "错误: 远端路径须为绝对路径: ${REMOTE_ABS}" >&2
        return 1
    fi
    if [[ "$(basename "${REMOTE_ABS}")" != "deploy_percept" ]]; then
        echo "错误: 远端 basename 须为 deploy_percept" >&2
        return 1
    fi
}

resolve_install_dir() {
    local preset="$1"
    local build_dir="${PROJECT_ROOT}/build/${preset}"
    if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
        echo "错误: 构建目录未配置: ${build_dir}" >&2
        return 1
    fi
    local prefix
    prefix="$(grep -E '^CMAKE_INSTALL_PREFIX:' "${build_dir}/CMakeCache.txt" | cut -d= -f2)"
    if [[ -z "${prefix}" || ! -d "${prefix}" ]]; then
        echo "错误: install prefix 不存在: ${prefix}" >&2
        return 1
    fi
    echo "${prefix}"
}

# 在本地跑 test_run.sh；把日志 tee 到文件；返回 exit code
run_local_dir() {
    local dir="$1"
    local log="$2"
    echo "=== test_run (local) ${dir} ==="
    set +e
    sh "${SCRIPT_DIR}/test_run.sh" "${dir}" 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    set -e
    return "${rc}"
}

# 在远端跑 test_run.sh；把日志 tee 到文件；返回 exit code
run_remote_dir() {
    local remote_dir="$1"
    local log="$2"
    echo "=== test_run (remote) ${remote_dir} ==="
    set +e
    "${SSH_WRAP[@]}" "${SSH_HOST}" \
        "sh $(printf '%q' "${REMOTE_ABS}/scripts/test_run.sh") $(printf '%q' "${remote_dir}")" \
        2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    set -e
    return "${rc}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset) PRESET="$2"; shift 2 ;;
        --board) BOARD="$2"; shift 2 ;;
        --full) DEPLOY_FULL=1; shift ;;
        --bench) RUN_BENCH=1; shift ;;
        --help) show_help; exit 0 ;;
        --profile)
            echo "错误: --profile 请另行执行（不经本脚本）" >&2
            exit 1
            ;;
        *)
            echo "错误: 未知参数: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

if [[ -n "${BOARD}" ]]; then
    echo "test: preset=${PRESET} mode=board board=${BOARD} full=${DEPLOY_FULL} bench=${RUN_BENCH}"
else
    echo "test: preset=${PRESET} mode=host bench=${RUN_BENCH}"
fi

bash "${SCRIPT_DIR}/build.sh" --preset "${PRESET}" --install
INSTALL_DIR="$(resolve_install_dir "${PRESET}")"

if [[ ! -d "${INSTALL_DIR}/tests" ]]; then
    echo "错误: 未找到 ${INSTALL_DIR}/tests" >&2
    exit 1
fi
if [[ "${RUN_BENCH}" -eq 1 && ! -d "${INSTALL_DIR}/benchmarks" ]]; then
    echo "错误: --bench 但未找到 ${INSTALL_DIR}/benchmarks（需 ENABLE_BENCHMARKS=ON）" >&2
    exit 1
fi

if [[ -n "${BOARD}" ]]; then
    ARTIFACTS="${PROJECT_ROOT}/ci-artifacts/board-logs"
    mkdir -p "${ARTIFACTS}"
    rm -rf "${ARTIFACTS:?}/"*

    setup_ssh_transport
    parse_board_spec "${BOARD}"

    sync_args=(--install-dir "${INSTALL_DIR}" --board "${BOARD}")
    if [[ "${DEPLOY_FULL}" -eq 1 ]]; then
        sync_args+=(--full)
    fi
    bash "${SCRIPT_DIR}/sync_install_to_board.sh" "${sync_args[@]}"

    {
        echo "board_spec=${BOARD}"
        echo "remote_abs=${REMOTE_ABS}"
        echo "deploy_full=${DEPLOY_FULL}"
        echo "bench=${RUN_BENCH}"
        echo "started_at=$(date -Iseconds)"
    } >"${ARTIFACTS}/meta.txt"

    rc=0
    run_remote_dir "${REMOTE_ABS}/tests" "${ARTIFACTS}/test_run.log" || rc=$?
    if [[ "${RUN_BENCH}" -eq 1 ]]; then
        run_remote_dir "${REMOTE_ABS}/benchmarks" "${ARTIFACTS}/benchmarks.log" || rc=$?
    fi
    echo "exit_code=${rc}" | tee -a "${ARTIFACTS}/meta.txt"
    echo "artifacts: ${ARTIFACTS}"
    exit "${rc}"
fi

ARTIFACTS="${PROJECT_ROOT}/ci-artifacts/host-logs"
mkdir -p "${ARTIFACTS}"
rm -rf "${ARTIFACTS:?}/"*
{
    echo "mode=host"
    echo "install_dir=${INSTALL_DIR}"
    echo "bench=${RUN_BENCH}"
    echo "started_at=$(date -Iseconds)"
} >"${ARTIFACTS}/meta.txt"

rc=0
run_local_dir "${INSTALL_DIR}/tests" "${ARTIFACTS}/test_run.log" || rc=$?
if [[ "${RUN_BENCH}" -eq 1 ]]; then
    run_local_dir "${INSTALL_DIR}/benchmarks" "${ARTIFACTS}/benchmarks.log" || rc=$?
fi
echo "exit_code=${rc}" | tee -a "${ARTIFACTS}/meta.txt"
echo "artifacts: ${ARTIFACTS}"
exit "${rc}"
