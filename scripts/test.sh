#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET="x86_64-debug"
BOARD=""
RUN_BENCH=0
BENCH_LOOPS=50

PERCEPT_TESTS_REL="share/percept/tests"
PERCEPT_BENCHMARKS_REL="share/percept/benchmarks"

PERCEPT_INSTALL_TESTS=(
    smoke_tests
    test_YoloV5DetectPostProcess
    test_YoloV5SegPostProcess
    test_YoloV8SegPostProcess
    test_yolov5_detect_awnn
    test_yolov8_detect_awnn
    test_mobilenetv2_classify_awnn
)

PERCEPT_INSTALL_BENCHMARKS=(
    bench_yolov5_post_process
    bench_yolov8_post_process
    bench_mobilenetv2_classify
)

SSH_WRAP=()
RSYNC_RSH=""

show_help() {
    echo "deploy_percept 测试脚本（build + install + 验证 install 包）"
    echo ""
    echo "用法: bash $0 --preset <name> [选项]"
    echo ""
    echo "流程:"
    echo "  PC 模式:     build → install → 本地 share/percept/tests/"
    echo "  开发板模式:  build → install → rsync → SSH 远程跑同一套测试"
    echo ""
    echo "选项:"
    echo "  --preset <name>          CMake preset（默认: x86_64-debug）"
    echo "  --board <user@host:path> 开发板 install 根目录（启用板端模式）"
    echo "  --bench [loops]          测试完成后再跑 benchmark（默认 loops=50）"
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

parse_board_spec() {
    local board_spec="$1"
    if [[ ! "${board_spec}" =~ ^([^@]+@[^:]+):(.+)$ ]]; then
        echo "错误: --board 格式应为 user@host:remote_path" >&2
        exit 1
    fi
    SSH_HOST="${BASH_REMATCH[1]}"
    REMOTE_DIR_INPUT="${BASH_REMATCH[2]}"
}

setup_install_tree_env() {
    local install_dir="$1"
    export PERCEPT_ROOT="${install_dir}/share/percept"
    export PERCEPT_OUTPUT_DIR="${install_dir}/var/percept/output"
    export LD_LIBRARY_PATH="${install_dir}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    mkdir -p "${PERCEPT_OUTPUT_DIR}"
}

run_install_tree_tests() {
    local install_dir="$1"
    local tests_dir="${install_dir}/${PERCEPT_TESTS_REL}"

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

run_install_tree_benchmarks() {
    local install_dir="$1"
    local loops="$2"
    local bench_dir="${install_dir}/${PERCEPT_BENCHMARKS_REL}"

    if [[ ! -d "${bench_dir}" ]]; then
        echo "错误: 未找到 benchmark 目录: ${bench_dir}" >&2
        echo "提示: ENABLE_BENCHMARKS=ON、INSTALL_BENCHMARKS=ON 且已 install" >&2
        return 1
    fi

    echo ""
    echo "=== install tree benchmarks (loops=${loops}) ==="
    echo "prefix:     ${install_dir}"
    echo "bench dir:  ${bench_dir}"

    local fail=0
    local b
    for b in "${PERCEPT_INSTALL_BENCHMARKS[@]}"; do
        if [[ ! -x "${bench_dir}/${b}" ]]; then
            echo "skip (not installed): ${b}"
            continue
        fi
        echo "=== ${b} ${loops} ==="
        "${bench_dir}/${b}" "${loops}" || fail=1
    done
    return "${fail}"
}

run_install_tree_suite() {
    local install_dir="$1"
    local loops="$2"

    if [[ ! -d "${install_dir}/share/percept" ]]; then
        echo "错误: 无效的 install 目录（缺少 share/percept/）: ${install_dir}" >&2
        exit 1
    fi
    install_dir="$(cd "${install_dir}" && pwd)"

    local tests_dir="${install_dir}/${PERCEPT_TESTS_REL}"
    if [[ ! -d "${tests_dir}" ]]; then
        echo "错误: 未找到测试目录: ${tests_dir}" >&2
        echo "提示: ENABLE_TESTS=ON、INSTALL_TESTS=ON" >&2
        exit 1
    fi

    setup_install_tree_env "${install_dir}"

    local fail=0
    run_install_tree_tests "${install_dir}" || fail=1

    if [[ "${RUN_BENCH}" -eq 1 ]]; then
        run_install_tree_benchmarks "${install_dir}" "${loops}" || fail=1
    fi

    return "${fail}"
}

run_rsync_to_board() {
    local install_dir="$1"
    local ssh_host="$2"
    local remote_dir_input="$3"

    echo ""
    echo "=== rsync install → board ==="
    echo "source:  ${install_dir}/"
    echo "target:  ${ssh_host}:${remote_dir_input}/"

    rsync -avz -e "${RSYNC_RSH}" "${install_dir}/" "${ssh_host}:${remote_dir_input}/"
}

run_install_tree_suite_remote() {
    local ssh_host="$1"
    local remote_dir_input="$2"
    local loops="$3"

    local remote_dir
    remote_dir="$("${SSH_WRAP[@]}" "${ssh_host}" "cd ${remote_dir_input} && pwd")" || {
        echo "错误: 无法访问远程目录: ${remote_dir_input}" >&2
        exit 1
    }

    local tests_dir="${remote_dir}/${PERCEPT_TESTS_REL}"
    local bench_dir="${remote_dir}/${PERCEPT_BENCHMARKS_REL}"

    echo ""
    echo "=== install tree tests (remote) ==="
    echo "host:      ${ssh_host}"
    echo "prefix:    ${remote_dir}"
    echo "tests dir: ${tests_dir}"
    if [[ "${RUN_BENCH}" -eq 1 ]]; then
        echo "bench dir: ${bench_dir} (loops=${loops})"
    fi

    local tests_shell=""
    local t
    for t in "${PERCEPT_INSTALL_TESTS[@]}"; do
        tests_shell+="if [[ -x \"${tests_dir}/${t}\" ]]; then echo '=== ${t} ==='; \"${tests_dir}/${t}\" || FAIL=1; else echo 'skip (not installed): ${t}'; fi;"
    done

    local bench_shell=""
    if [[ "${RUN_BENCH}" -eq 1 ]]; then
        bench_shell="
echo ''
echo '=== install tree benchmarks (remote, loops=${loops}) ==='
if [[ ! -d \"${bench_dir}\" ]]; then
  echo \"错误: 未找到 benchmark 目录: ${bench_dir}\" >&2
  FAIL=1
fi
"
        local b
        for b in "${PERCEPT_INSTALL_BENCHMARKS[@]}"; do
            bench_shell+="if [[ -x \"${bench_dir}/${b}\" ]]; then echo '=== ${b} ${loops} ==='; \"${bench_dir}/${b}\" ${loops} || FAIL=1; else echo 'skip (not installed): ${b}'; fi;"
        done
    fi

    "${SSH_WRAP[@]}" "${ssh_host}" "set -euo pipefail
FAIL=0
export PERCEPT_ROOT=\"${remote_dir}/share/percept\"
export PERCEPT_OUTPUT_DIR=\"${remote_dir}/var/percept/output\"
export LD_LIBRARY_PATH=\"${remote_dir}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"
mkdir -p \"\${PERCEPT_OUTPUT_DIR}\"
if [[ ! -d \"${tests_dir}\" ]]; then
  echo \"错误: 未找到测试目录: ${tests_dir}\" >&2
  echo \"提示: INSTALL_TESTS=ON 且已 install + rsync\" >&2
  exit 1
fi
${tests_shell}
${bench_shell}
exit \${FAIL}"
}

run_board_suite() {
    local board_spec="$1"
    local install_dir="$2"
    local loops="$3"

    parse_board_spec "${board_spec}"
    setup_ssh_transport
    run_rsync_to_board "${install_dir}" "${SSH_HOST}" "${REMOTE_DIR_INPUT}"
    run_install_tree_suite_remote "${SSH_HOST}" "${REMOTE_DIR_INPUT}" "${loops}"
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
    run_board_suite "${BOARD}" "${INSTALL_DIR}" "${BENCH_LOOPS}"
    exit $?
fi

run_install_tree_suite "${INSTALL_DIR}" "${BENCH_LOOPS}"
exit $?
