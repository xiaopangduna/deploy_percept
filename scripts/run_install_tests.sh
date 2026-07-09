#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="$(cd "${PERCEPT_INSTALL_PREFIX:-${SCRIPT_DIR}/..}" && pwd)"

RUN_BENCH=0
BENCH_LOOPS=50

TESTS_DIR="${INSTALL_PREFIX}/tests"
BENCHMARKS_DIR="${INSTALL_PREFIX}/benchmarks"

show_help() {
    echo "deploy_percept install 包测试 runner"
    echo ""
    echo "用法: ${INSTALL_PREFIX}/scripts/run_install_tests.sh [选项]"
    echo ""
    echo "默认: 扫描并运行 tests/ 下全部可执行文件"
    echo ""
    echo "选项:"
    echo "  --bench [loops]   tests 完成后再跑 benchmarks/（默认 loops=50）"
    echo "  --help            显示帮助"
    echo ""
    echo "环境变量:"
    echo "  PERCEPT_INSTALL_PREFIX  覆盖 install 根目录（默认: 脚本上级目录）"
}

is_unsigned_loops() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --bench)
                RUN_BENCH=1
                shift
                if [[ $# -gt 0 ]] && is_unsigned_loops "$1"; then
                    BENCH_LOOPS="$1"
                    shift
                fi
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "错误: 未知参数: $1" >&2
                show_help
                exit 1
                ;;
        esac
    done
}

validate_install_tree() {
    if [[ ! -d "${INSTALL_PREFIX}/share/percept" ]]; then
        echo "错误: 无效的 install 目录（缺少 share/percept/）: ${INSTALL_PREFIX}" >&2
        exit 1
    fi
    if [[ ! -d "${TESTS_DIR}" ]]; then
        echo "错误: 未找到测试目录: ${TESTS_DIR}" >&2
        echo "提示: ENABLE_TESTS=ON、INSTALL_TESTS=ON 且已 install" >&2
        exit 1
    fi
}

setup_install_tree_env() {
    export PERCEPT_ROOT="${INSTALL_PREFIX}/share/percept"
    export PERCEPT_OUTPUT_DIR="${INSTALL_PREFIX}/var/percept/output"
    export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    mkdir -p "${PERCEPT_OUTPUT_DIR}"
}

discover_executables() {
    local dir="$1"
    find "${dir}" -maxdepth 1 -type f -executable -printf '%f\n' 2>/dev/null | sort
}

run_executables_in_dir() {
    local dir="$1"
    local label="$2"
    shift 2
    local extra_args=("$@")

    local names
    mapfile -t names < <(discover_executables "${dir}")

    echo "=== ${label} ==="
    echo "prefix: ${INSTALL_PREFIX}"
    echo "dir:    ${dir}"

    if [[ ${#names[@]} -eq 0 ]]; then
        echo "skip: 目录下无可执行文件"
        return 0
    fi

    local fail=0
    local name
    for name in "${names[@]}"; do
        echo "=== ${name} ==="
        if [[ ${#extra_args[@]} -gt 0 ]]; then
            "${dir}/${name}" "${extra_args[@]}" || fail=1
        else
            "${dir}/${name}" || fail=1
        fi
    done
    return "${fail}"
}

run_benchmarks() {
    if [[ ! -d "${BENCHMARKS_DIR}" ]]; then
        echo "错误: 未找到 benchmark 目录: ${BENCHMARKS_DIR}" >&2
        echo "提示: ENABLE_BENCHMARKS=ON、INSTALL_BENCHMARKS=ON 且已 install 到 benchmarks/" >&2
        return 1
    fi

    echo ""
    run_executables_in_dir "${BENCHMARKS_DIR}" "install tree benchmarks (loops=${BENCH_LOOPS})" "${BENCH_LOOPS}"
}

main() {
    parse_args "$@"

    validate_install_tree
    setup_install_tree_env

    local fail=0
    run_executables_in_dir "${TESTS_DIR}" "install tree tests" || fail=1

    if [[ "${RUN_BENCH}" -eq 1 ]]; then
        run_benchmarks || fail=1
    fi

    exit "${fail}"
}

main "$@"
