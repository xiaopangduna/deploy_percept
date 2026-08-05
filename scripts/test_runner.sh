#!/bin/sh
# busybox ash / POSIX sh 兼容（Sigmastar 等板无 bash）
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_PREFIX="$(cd "${PERCEPT_INSTALL_PREFIX:-${SCRIPT_DIR}/..}" && pwd)"

RUN_BENCH=0
BENCH_LOOPS=50
RUN_PROFILE=0
PROFILE_DURATION_SEC=60

TESTS_DIR="${INSTALL_PREFIX}/tests"
BENCHMARKS_DIR="${INSTALL_PREFIX}/benchmarks"
PROFILING_DIR="${INSTALL_PREFIX}/profiling"
BIN_DIR="${INSTALL_PREFIX}/bin"
FIXTURES_ROOT="${INSTALL_PREFIX}/share/percept/fixtures"
LOGS_DIR="${INSTALL_PREFIX}/var/percept/logs"

show_help() {
    echo "deploy_percept install 包测试 runner"
    echo ""
    echo "用法: ${INSTALL_PREFIX}/scripts/test_runner.sh [选项]"
    echo ""
    echo "默认: 扫描并运行 tests/ 下全部可执行文件"
    echo ""
    echo "选项:"
    echo "  --bench [loops]     tests 完成后再跑 benchmarks/（默认 loops=50）"
    echo "  --profile [sec]     tests 后再跑资源画像；sec=app 循环墙钟秒数（默认 60）"
    echo "  --help              显示帮助"
    echo ""
    echo "环境变量:"
    echo "  PERCEPT_INSTALL_PREFIX  覆盖 install 根目录（默认: 脚本上级目录）"
    echo ""
    echo "日志目录: ${INSTALL_PREFIX}/var/percept/logs/"
    echo "  test.log  bench.log  profile.log  profile.csv  meta.txt"
}

is_unsigned() {
    case "${1:-}" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --bench)
                RUN_BENCH=1
                shift
                if [ $# -gt 0 ] && is_unsigned "$1"; then
                    BENCH_LOOPS="$1"
                    shift
                fi
                ;;
            --profile)
                RUN_PROFILE=1
                shift
                if [ $# -gt 0 ] && is_unsigned "$1"; then
                    PROFILE_DURATION_SEC="$1"
                    shift
                fi
                if [ "${PROFILE_DURATION_SEC}" -eq 0 ]; then
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
    if [ ! -d "${INSTALL_PREFIX}/share/percept" ]; then
        echo "错误: 无效的 install 目录（缺少 share/percept/）: ${INSTALL_PREFIX}" >&2
        exit 1
    fi
    if [ ! -d "${TESTS_DIR}" ]; then
        echo "错误: 未找到测试目录: ${TESTS_DIR}" >&2
        echo "提示: ENABLE_TESTS=ON、INSTALL_TESTS=ON 且已 install" >&2
        exit 1
    fi
}

setup_install_tree_env() {
    export PERCEPT_ROOT="${INSTALL_PREFIX}/share/percept"
    export PERCEPT_OUTPUT_DIR="${INSTALL_PREFIX}/var/percept/output"
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib"
    fi
    mkdir -p "${PERCEPT_OUTPUT_DIR}"
    mkdir -p "${LOGS_DIR}"
    # 每次 runner 覆盖固定日志名，便于板端回传与版本对比
    rm -f "${LOGS_DIR}/test.log" \
        "${LOGS_DIR}/bench.log" \
        "${LOGS_DIR}/profile.log" \
        "${LOGS_DIR}/profile.csv" \
        "${LOGS_DIR}/meta.txt" 2>/dev/null || true
}

write_meta() {
    {
        if date -Iseconds >/dev/null 2>&1; then
            echo "timestamp=$(date -Iseconds)"
        else
            echo "timestamp=$(date)"
        fi
        echo "hostname=$(hostname 2>/dev/null || echo unknown)"
        echo "uid=$(id -u)"
        echo "user=$(id -un 2>/dev/null || echo unknown)"
        echo "install_prefix=${INSTALL_PREFIX}"
        echo "run_bench=${RUN_BENCH}"
        echo "bench_loops=${BENCH_LOOPS}"
        echo "run_profile=${RUN_PROFILE}"
        echo "profile_duration_sec=${PROFILE_DURATION_SEC}"
    } >"${LOGS_DIR}/meta.txt"
}

# 列出目录下可执行文件名（兼容 busybox find，无 -printf/-executable）
discover_executables() {
    _dir="$1"
    find "${_dir}" -maxdepth 1 -type f 2>/dev/null | while IFS= read -r _f; do
        [ -x "${_f}" ] || continue
        basename "${_f}"
    done | sort
}

# 将目录下可执行文件依次跑完；可选第 4 参作为传给每个可执行文件的单一参数（如 bench loops）
run_executables_in_dir() {
    _dir="$1"
    _label="$2"
    _log_file="$3"
    _extra_arg="${4:-}"

    {
        echo "=== ${_label} ==="
        echo "prefix: ${INSTALL_PREFIX}"
        echo "dir:    ${_dir}"
        echo ""
    } | tee -a "${_log_file}"

    _names_file="$(mktemp)"
    discover_executables "${_dir}" >"${_names_file}" || true

    if [ ! -s "${_names_file}" ]; then
        echo "skip: 目录下无可执行文件" | tee -a "${_log_file}"
        rm -f "${_names_file}"
        return 0
    fi

    _fail=0
    while IFS= read -r _name; do
        [ -n "${_name}" ] || continue
        {
            echo "========== ${_name} =========="
        } | tee -a "${_log_file}"

        _out_tmp="$(mktemp)"
        set +e
        if [ -n "${_extra_arg}" ]; then
            "${_dir}/${_name}" "${_extra_arg}" >"${_out_tmp}" 2>&1
        else
            "${_dir}/${_name}" >"${_out_tmp}" 2>&1
        fi
        _rc=$?
        set -e
        tee -a "${_log_file}" <"${_out_tmp}"
        rm -f "${_out_tmp}"
        if [ "${_rc}" -ne 0 ]; then
            _fail=1
        fi
        echo "" | tee -a "${_log_file}"
    done <"${_names_file}"
    rm -f "${_names_file}"
    return "${_fail}"
}

run_benchmarks() {
    if [ ! -d "${BENCHMARKS_DIR}" ]; then
        echo "错误: 未找到 benchmark 目录: ${BENCHMARKS_DIR}" >&2
        echo "提示: ENABLE_BENCHMARKS=ON、INSTALL_BENCHMARKS=ON 且已 install 到 benchmarks/" >&2
        return 1
    fi

    echo ""
    run_executables_in_dir \
        "${BENCHMARKS_DIR}" \
        "install tree benchmarks (loops=${BENCH_LOOPS})" \
        "${LOGS_DIR}/bench.log" \
        "${BENCH_LOOPS}"
}

# 已知支持 --duration-sec 的 app：name|相对 fixture 模型|相对 fixture 输入
profile_targets() {
    cat <<'EOF'
mobilenetv2_classify_awnn|integration/awnn/mobilenetv2_classify/mobilenetv2.nb|integration/awnn/mobilenetv2_classify/cat.jpg
EOF
}

run_one_profile_target() {
    _app_name="$1"
    _model_rel="$2"
    _input_rel="$3"

    _app_bin="${BIN_DIR}/${_app_name}"
    _monitor_bin="${PROFILING_DIR}/percept_monitor"
    _model_path="${FIXTURES_ROOT}/${_model_rel}"
    _input_path="${FIXTURES_ROOT}/${_input_rel}"

    if [ ! -x "${_app_bin}" ]; then
        echo "skip: 未找到 ${_app_bin}"
        return 0
    fi
    if [ ! -f "${_model_path}" ]; then
        echo "错误: fixture 模型不存在: ${_model_path}" >&2
        return 1
    fi
    if [ ! -f "${_input_path}" ]; then
        echo "错误: fixture 输入不存在: ${_input_path}" >&2
        return 1
    fi

    echo "=== profile ${_app_name} (duration-sec=${PROFILE_DURATION_SEC}) ==="
    echo "app:     ${_app_bin}"
    echo "monitor: ${_monitor_bin}"
    echo "model:   ${_model_path}"
    echo "input:   ${_input_path}"
    if [ "$(id -u)" -ne 0 ]; then
        echo "提示: 当前非 root，NPU core_loading 可能无法读取；板端建议 --board root@host:path" >&2
    fi

    _profile_log="${LOGS_DIR}/profile.log"
    _profile_csv="${LOGS_DIR}/profile.csv"
    _app_tmp="$(mktemp)"

    {
        echo "========== profile ${_app_name} =========="
        echo "duration-sec=${PROFILE_DURATION_SEC}"
        echo "model=${_model_path}"
        echo "input=${_input_path}"
        echo ""
    } | tee -a "${_profile_log}"

    "${_app_bin}" --duration-sec "${PROFILE_DURATION_SEC}" \
        "${_model_path}" "${_input_path}" >"${_app_tmp}" 2>&1 &
    _app_pid=$!

    _monitor_rc=0
    _app_rc=0
    {
        echo "--- percept_monitor ---"
    } | tee -a "${_profile_log}"

    _mon_tmp="$(mktemp)"
    set +e
    "${_monitor_bin}" --name "${_app_name}" --out "${_profile_csv}" >"${_mon_tmp}" 2>&1
    _monitor_rc=$?
    set -e
    tee -a "${_profile_log}" <"${_mon_tmp}"
    rm -f "${_mon_tmp}"

    set +e
    wait "${_app_pid}"
    _app_rc=$?
    set -e

    {
        echo ""
        echo "--- ${_app_name} app ---"
        cat "${_app_tmp}"
        echo ""
        echo "--- profile.csv: ${_profile_csv} ---"
        echo ""
    } | tee -a "${_profile_log}"
    rm -f "${_app_tmp}"

    if [ "${_app_rc}" -ne 0 ]; then
        echo "错误: ${_app_name} 退出码非 0" >&2
        return 1
    fi
    if [ "${_monitor_rc}" -ne 0 ]; then
        echo "错误: percept_monitor 失败 (${_app_name})" >&2
        return 1
    fi
    return 0
}

run_profiling() {
    _monitor_bin="${PROFILING_DIR}/percept_monitor"
    if [ ! -x "${_monitor_bin}" ]; then
        echo "错误: 未找到 ${_monitor_bin}" >&2
        echo "提示: ENABLE_PROFILING=ON、INSTALL_PROFILING=ON 且已 install" >&2
        return 1
    fi

    echo ""
    echo "=== install tree profiling (duration-sec=${PROFILE_DURATION_SEC}) ==="
    echo "prefix: ${INSTALL_PREFIX}"

    _targets_file="$(mktemp)"
    profile_targets >"${_targets_file}"

    _ran=0
    _fail=0
    while IFS='|' read -r _app_name _model_rel _input_rel; do
        [ -n "${_app_name}" ] || continue
        if [ ! -x "${BIN_DIR}/${_app_name}" ]; then
            echo "skip: ${_app_name}（bin 中无此 app）"
            continue
        fi
        _ran=1
        if ! run_one_profile_target "${_app_name}" "${_model_rel}" "${_input_rel}"; then
            _fail=1
        fi
    done <"${_targets_file}"
    rm -f "${_targets_file}"

    if [ "${_ran}" -eq 0 ]; then
        echo "skip: 无可用 profiling 目标（例如板端 AWNN app 未安装）"
        return 0
    fi
    return "${_fail}"
}

main() {
    parse_args "$@"

    validate_install_tree
    setup_install_tree_env
    write_meta

    echo "logs: ${LOGS_DIR}"

    _fail=0
    if ! run_executables_in_dir "${TESTS_DIR}" "install tree tests" "${LOGS_DIR}/test.log"; then
        _fail=1
    fi

    if [ "${RUN_BENCH}" -eq 1 ]; then
        if ! run_benchmarks; then
            _fail=1
        fi
    fi

    if [ "${RUN_PROFILE}" -eq 1 ]; then
        if ! run_profiling; then
            _fail=1
        fi
    fi

    {
        echo "logs=test.log,bench.log,profile.log,profile.csv,meta.txt"
        ls -1 "${LOGS_DIR}" 2>/dev/null || true
    } >>"${LOGS_DIR}/meta.txt"

    echo "logs written under: ${LOGS_DIR}"
    exit "${_fail}"
}

main "$@"
