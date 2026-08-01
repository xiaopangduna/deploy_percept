#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="$(cd "${PERCEPT_INSTALL_PREFIX:-${SCRIPT_DIR}/..}" && pwd)"

RUN_BENCH=0
BENCH_LOOPS=50
RUN_PROFILE=0
PROFILE_DURATION_SEC=30

TESTS_DIR="${INSTALL_PREFIX}/tests"
BENCHMARKS_DIR="${INSTALL_PREFIX}/benchmarks"
PROFILING_DIR="${INSTALL_PREFIX}/profiling"
BIN_DIR="${INSTALL_PREFIX}/bin"
FIXTURES_ROOT="${INSTALL_PREFIX}/share/percept/fixtures"

show_help() {
    echo "deploy_percept install 包测试 runner"
    echo ""
    echo "用法: ${INSTALL_PREFIX}/scripts/test_runner.sh [选项]"
    echo ""
    echo "默认: 扫描并运行 tests/ 下全部可执行文件"
    echo ""
    echo "选项:"
    echo "  --bench [loops]     tests 完成后再跑 benchmarks/（默认 loops=50）"
    echo "  --profile [sec]     tests 后再跑资源画像；sec=app 循环墙钟秒数（默认 30）"
    echo "  --help              显示帮助"
    echo ""
    echo "环境变量:"
    echo "  PERCEPT_INSTALL_PREFIX  覆盖 install 根目录（默认: 脚本上级目录）"
}

is_unsigned() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
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

# 已知支持 --duration-sec 的 app：name|相对 fixture 模型|相对 fixture 输入
profile_targets() {
    cat <<'EOF'
mobilenetv2_classify_awnn|integration/awnn/mobilenetv2_classify/mobilenetv2.nb|integration/awnn/mobilenetv2_classify/cat.jpg
EOF
}

run_one_profile_target() {
    local app_name="$1"
    local model_rel="$2"
    local input_rel="$3"

    local app_bin="${BIN_DIR}/${app_name}"
    local monitor_bin="${PROFILING_DIR}/percept_monitor"
    local model_path="${FIXTURES_ROOT}/${model_rel}"
    local input_path="${FIXTURES_ROOT}/${input_rel}"

    if [[ ! -x "${app_bin}" ]]; then
        echo "skip: 未找到 ${app_bin}"
        return 0
    fi
    if [[ ! -f "${model_path}" ]]; then
        echo "错误: fixture 模型不存在: ${model_path}" >&2
        return 1
    fi
    if [[ ! -f "${input_path}" ]]; then
        echo "错误: fixture 输入不存在: ${input_path}" >&2
        return 1
    fi

    echo "=== profile ${app_name} (duration-sec=${PROFILE_DURATION_SEC}) ==="
    echo "app:     ${app_bin}"
    echo "monitor: ${monitor_bin}"
    echo "model:   ${model_path}"
    echo "input:   ${input_path}"

    local app_log
    app_log="$(mktemp "${PERCEPT_OUTPUT_DIR}/profile_${app_name}.XXXXXX.log")"

    "${app_bin}" --duration-sec "${PROFILE_DURATION_SEC}" \
        "${model_path}" "${input_path}" >"${app_log}" 2>&1 &
    local app_pid=$!

    local monitor_rc=0
    local app_rc=0
    if ! "${monitor_bin}" --name "${app_name}"; then
        monitor_rc=1
    fi

    if ! wait "${app_pid}"; then
        app_rc=1
    fi

    echo "--- ${app_name} app log: ${app_log} ---"
    cat "${app_log}" || true

    if [[ "${app_rc}" -ne 0 ]]; then
        echo "错误: ${app_name} 退出码非 0" >&2
        return 1
    fi
    if [[ "${monitor_rc}" -ne 0 ]]; then
        echo "错误: percept_monitor 失败 (${app_name})" >&2
        return 1
    fi
    return 0
}

run_profiling() {
    local monitor_bin="${PROFILING_DIR}/percept_monitor"
    if [[ ! -x "${monitor_bin}" ]]; then
        echo "错误: 未找到 ${monitor_bin}" >&2
        echo "提示: ENABLE_PROFILING=ON、INSTALL_PROFILING=ON 且已 install" >&2
        return 1
    fi

    echo ""
    echo "=== install tree profiling (duration-sec=${PROFILE_DURATION_SEC}) ==="
    echo "prefix: ${INSTALL_PREFIX}"

    local ran=0
    local fail=0
    local line app_name model_rel input_rel
    while IFS='|' read -r app_name model_rel input_rel; do
        [[ -z "${app_name}" ]] && continue
        if [[ ! -x "${BIN_DIR}/${app_name}" ]]; then
            echo "skip: ${app_name}（bin 中无此 app）"
            continue
        fi
        ran=1
        run_one_profile_target "${app_name}" "${model_rel}" "${input_rel}" || fail=1
    done < <(profile_targets)

    if [[ "${ran}" -eq 0 ]]; then
        echo "skip: 无可用 profiling 目标（例如板端 AWNN app 未安装）"
        return 0
    fi
    return "${fail}"
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

    if [[ "${RUN_PROFILE}" -eq 1 ]]; then
        run_profiling || fail=1
    fi

    exit "${fail}"
}

main "$@"
