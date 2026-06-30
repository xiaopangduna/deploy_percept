#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET="x86_64-debug"
INSTALL_DIR=""
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
)

PERCEPT_INSTALL_BENCHMARKS=(
    bench_yolov5_post_process
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
    echo "  bash $0 --install-dir <path> [--bench [loops]]"
    echo ""
    echo "模式 C — 开发板（install 已 rsync 到板子）:"
    echo "  bash $0 --board <user@host:remote_path> [--bench [loops]]"
    echo ""
    echo "选项:"
    echo "  --preset <name>            CMake preset（默认: x86_64-debug，用于 ctest）"
    echo "  --install-dir <path>       对 install prefix 跑 share/percept/tests/ 内测试"
    echo "  --board <user@host:path>   SSH 到开发板跑 install tree 测试"
    echo "  --bench [loops]            install/板端模式下，测试完成后再跑 benchmark（默认 loops=50）"
    echo "  --help                     显示帮助"
    echo ""
    echo "示例:"
    echo "  bash scripts/build.sh --preset x86_64-debug"
    echo "  bash $0 --preset x86_64-debug"
    echo "  bash $0 --install-dir install/x86_64"
    echo "  bash $0 --board orangepi@192.168.1.10:~/deploy_percept"
    echo "  bash $0 --board orangepi@192.168.1.10:~/deploy_percept --bench"
    echo "  bash $0 --board orangepi@192.168.1.10:~/deploy_percept --bench 100"
}

is_unsigned_loops() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
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
        echo "提示: ENABLE_BENCHMARKS=ON、INSTALL_BENCHMARKS=ON 且已 cmake --install" >&2
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
        echo "提示: 构建时需 ENABLE_TESTS=ON 且 INSTALL_TESTS=ON，并执行 cmake --install" >&2
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

run_install_tree_suite_remote() {
    local board_spec="$1"
    local loops="$2"

    if [[ ! "${board_spec}" =~ ^([^@]+@[^:]+):(.+)$ ]]; then
        echo "错误: --board 格式应为 user@host:remote_path" >&2
        exit 1
    fi
    local ssh_host="${BASH_REMATCH[1]}"
    local remote_dir_input="${BASH_REMATCH[2]}"

    local remote_dir
    remote_dir="$(ssh "${ssh_host}" "cd ${remote_dir_input} && pwd")" || {
        echo "错误: 无法访问远程目录: ${remote_dir_input}" >&2
        exit 1
    }

    local tests_dir="${remote_dir}/${PERCEPT_TESTS_REL}"
    local bench_dir="${remote_dir}/${PERCEPT_BENCHMARKS_REL}"

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
${bench_shell}
exit \${FAIL}"
}

run_ctest() {
    local build_dir="${PROJECT_ROOT}/build/${PRESET}"
    if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
        echo "错误: 构建目录未配置: ${build_dir}" >&2
        echo "请先执行: bash scripts/build.sh --preset ${PRESET}" >&2
        exit 1
    fi
    if [[ "${RUN_BENCH}" -eq 1 ]]; then
        echo "提示: --bench 仅用于 --install-dir / --board，ctest 模式下已忽略" >&2
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
    run_install_tree_suite_remote "${BOARD}" "${BENCH_LOOPS}"
    exit $?
fi

if [[ -n "${INSTALL_DIR}" ]]; then
    if [[ "${INSTALL_DIR}" != /* ]]; then
        INSTALL_DIR="${PROJECT_ROOT}/${INSTALL_DIR}"
    fi
    run_install_tree_suite "${INSTALL_DIR}" "${BENCH_LOOPS}"
    exit $?
fi

run_ctest
