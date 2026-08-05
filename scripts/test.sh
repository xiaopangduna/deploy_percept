#!/bin/bash
# 主机编排：discover →（可选 scp）→ 逐条 exec → 汇总日志
# 板端只跑 ELF；环境变量由本脚本拼进命令，不依赖 board_env.sh / 板上复杂 runner。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET="x86_64-debug"
BOARD=""
RUN_BENCH=0
BENCH_LOOPS=50
RUN_PROFILE=0
PROFILE_DURATION_SEC=60
LEGACY_RUNNER=0

SSH_WRAP=()
SCP_WRAP=()

show_help() {
    echo "deploy_percept 测试脚本（主机编排：build+install → 逐条跑 tests）"
    echo ""
    echo "用法: bash $0 [选项]"
    echo ""
    echo "默认 host: 本地 install，主机发现并执行 tests/"
    echo "--board:   远程清空后 scp，主机逐条 ssh 执行（板端无需 bash/rsync）"
    echo ""
    echo "选项:"
    echo "  --preset <name>          CMake preset（默认: x86_64-debug）"
    echo "  --board <user@host:path> 板端模式（user@host:远程 install 根目录）"
    echo "  --bench [loops]          再跑 benchmarks/（默认 loops=50）"
    echo "  --profile [sec]          再跑 profiling（默认 60s）"
    echo "  --legacy-runner          仍调用板上/本地 scripts/test_runner.sh（过渡）"
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
    # -n：stdin 接 /dev/null，避免 while-read 循环里 ssh 吞掉后续测试名
    SSH_WRAP=(sshpass -e ssh -n -o StrictHostKeyChecking=accept-new)
    SCP_WRAP=(sshpass -e scp -o StrictHostKeyChecking=accept-new)
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

# 列出目录下可执行文件名（主机侧）
discover_executables() {
    local dir="$1"
    if [[ ! -d "${dir}" ]]; then
        return 0
    fi
    find "${dir}" -maxdepth 1 -type f -executable -printf '%f\n' 2>/dev/null | sort
}

# 在 prefix 下执行相对路径命令所需的环境（内联进 local/ssh，不落板脚本）
prefix_env_exports() {
    local prefix="$1"
    # 仅使用 install 树内 lib（勿把主机 LD_LIBRARY_PATH 拼进远程命令）
    printf 'export LD_LIBRARY_PATH=%q; ' "${prefix}/lib"
    printf 'export PERCEPT_ROOT=%q; ' "${prefix}/share/percept"
    printf 'export PERCEPT_OUTPUT_DIR=%q; ' "${prefix}/var/percept/output"
    printf 'mkdir -p %q; ' "${prefix}/var/percept/output"
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

run_legacy_runner_local() {
    local install_dir="$1"
    local runner="${install_dir}/scripts/test_runner.sh"
    if [[ ! -x "${runner}" ]]; then
        echo "错误: 未找到 runner: ${runner}" >&2
        exit 1
    fi
    build_runner_args
    "${runner}" "${RUNNER_ARGS[@]}"
}

remote_clear_dir() {
    local ssh_host="$1"
    local remote_abs="$2"
    if [[ -z "${remote_abs}" || "${remote_abs}" == "/" || "${remote_abs}" == "/root" ||
          "${remote_abs}" == "/home" || "${remote_abs}" == "/customer" ]]; then
        echo "错误: 拒绝清空可疑远程路径: ${remote_abs}" >&2
        exit 1
    fi
    "${SSH_WRAP[@]}" "${ssh_host}" \
        "mkdir -p $(printf '%q' "${remote_abs}") && cd $(printf '%q' "${remote_abs}") && rm -rf ./* ./.[!.]* ./..?* 2>/dev/null || true"
}

scp_install_tree() {
    local install_dir="$1"
    local ssh_host="$2"
    local remote_abs="$3"
    local -a entries=()

    echo "=== clear remote ${ssh_host}:${remote_abs}/ ==="
    remote_clear_dir "${ssh_host}" "${remote_abs}"

    echo "=== scp → ${ssh_host}:${remote_abs}/ ==="
    shopt -s nullglob dotglob
    entries=("${install_dir}"/*)
    shopt -u nullglob dotglob
    if [[ ${#entries[@]} -eq 0 ]]; then
        echo "错误: install 目录为空: ${install_dir}" >&2
        exit 1
    fi
    "${SCP_WRAP[@]}" -r "${entries[@]}" "${ssh_host}:${remote_abs}/"
}

# 本地执行：cwd=prefix，相对路径 rel（如 tests/test_smoke）
run_one_local() {
    local prefix="$1"
    local rel="$2"
    local log_file="$3"
    shift 3
    local -a extra=("$@")

    mkdir -p "$(dirname "${log_file}")"
    {
        echo "========== ${rel} =========="
        echo "executor=local"
        echo "prefix=${prefix}"
        echo ""
    } >"${log_file}"

    set +e
    (
        cd "${prefix}"
        # shellcheck disable=SC1090
        eval "$(prefix_env_exports "${prefix}")"
        if [[ ${#extra[@]} -gt 0 ]]; then
            "./${rel}" "${extra[@]}"
        else
            "./${rel}"
        fi
    ) >>"${log_file}" 2>&1
    local rc=$?
    set -e
    echo "exit_code=${rc}" >>"${log_file}"
    return "${rc}"
}

# 远程执行：ssh + 内联 env
run_one_remote() {
    local ssh_host="$1"
    local remote_abs="$2"
    local rel="$3"
    local log_file="$4"
    shift 4
    local -a extra=("$@")

    mkdir -p "$(dirname "${log_file}")"
    {
        echo "========== ${rel} =========="
        echo "executor=remote"
        echo "host=${ssh_host}"
        echo "prefix=${remote_abs}"
        echo ""
    } >"${log_file}"

    local remote_cmd=""
    remote_cmd+="cd $(printf '%q' "${remote_abs}") && "
    remote_cmd+="$(prefix_env_exports "${remote_abs}")"
    remote_cmd+="./$(printf '%q' "${rel}")"
    local a
    for a in "${extra[@]}"; do
        remote_cmd+=" $(printf '%q' "${a}")"
    done

    set +e
    "${SSH_WRAP[@]}" "${ssh_host}" "${remote_cmd}" >>"${log_file}" 2>&1
    local rc=$?
    set -e
    echo "exit_code=${rc}" >>"${log_file}"
    return "${rc}"
}

# profile：单次 ssh 内 app 后台 + monitor（逻辑从旧 runner 上收）
run_one_profile_remote() {
    local ssh_host="$1"
    local remote_abs="$2"
    local app_name="$3"
    local model_rel="$4"
    local input_rel="$5"
    local log_file="$6"
    local csv_file="$7"

    mkdir -p "$(dirname "${log_file}")" "$(dirname "${csv_file}")"
    local remote_csv="${remote_abs}/var/percept/output/profile_${app_name}.csv"
    local model_path="${remote_abs}/share/percept/fixtures/${model_rel}"
    local input_path="${remote_abs}/share/percept/fixtures/${input_rel}"

    {
        echo "========== profile ${app_name} =========="
        echo "duration-sec=${PROFILE_DURATION_SEC}"
        echo "model=${model_path}"
        echo "input=${input_path}"
        echo ""
    } >"${log_file}"

    # 板上用 sh -c；busybox 无 bash
    local remote_script
    remote_script=$(
        cat <<EOF
set -eu
cd $(printf '%q' "${remote_abs}")
$(prefix_env_exports "${remote_abs}")
APP=./bin/$(printf '%q' "${app_name}")
MON=./profiling/percept_monitor
MODEL=$(printf '%q' "${model_path}")
INPUT=$(printf '%q' "${input_path}")
CSV=$(printf '%q' "${remote_csv}")
DUR=$(printf '%q' "${PROFILE_DURATION_SEC}")
if [ ! -x "\$APP" ]; then echo "skip: missing \$APP"; exit 0; fi
if [ ! -x "\$MON" ]; then echo "error: missing \$MON"; exit 1; fi
if [ ! -f "\$MODEL" ] || [ ! -f "\$INPUT" ]; then echo "error: missing fixture"; exit 1; fi
APP_TMP=\$(mktemp)
\$APP --duration-sec "\$DUR" "\$MODEL" "\$INPUT" >"\$APP_TMP" 2>&1 &
APP_PID=\$!
set +e
\$MON --name $(printf '%q' "${app_name}") --out "\$CSV"
MON_RC=\$?
wait "\$APP_PID"
APP_RC=\$?
set -e
echo "--- app ---"
cat "\$APP_TMP"
rm -f "\$APP_TMP"
echo "app_rc=\$APP_RC mon_rc=\$MON_RC"
[ "\$APP_RC" -eq 0 ] && [ "\$MON_RC" -eq 0 ]
EOF
    )

    set +e
    "${SSH_WRAP[@]}" "${ssh_host}" "sh -c $(printf '%q' "${remote_script}")" >>"${log_file}" 2>&1
    local rc=$?
    set -e
    echo "exit_code=${rc}" >>"${log_file}"

    if "${SSH_WRAP[@]}" "${ssh_host}" "test -f $(printf '%q' "${remote_csv}")"; then
        "${SCP_WRAP[@]}" "${ssh_host}:${remote_csv}" "${csv_file}" 2>/dev/null || true
    fi
    return "${rc}"
}

run_one_profile_local() {
    local prefix="$1"
    local app_name="$2"
    local model_rel="$3"
    local input_rel="$4"
    local log_file="$5"
    local csv_file="$6"

    local app_bin="${prefix}/bin/${app_name}"
    local mon_bin="${prefix}/profiling/percept_monitor"
    local model_path="${prefix}/share/percept/fixtures/${model_rel}"
    local input_path="${prefix}/share/percept/fixtures/${input_rel}"

    mkdir -p "$(dirname "${log_file}")" "$(dirname "${csv_file}")"
    {
        echo "========== profile ${app_name} =========="
        echo "duration-sec=${PROFILE_DURATION_SEC}"
        echo ""
    } >"${log_file}"

    if [[ ! -x "${app_bin}" ]]; then
        echo "skip: missing ${app_bin}" | tee -a "${log_file}"
        return 0
    fi
    if [[ ! -x "${mon_bin}" ]]; then
        echo "error: missing ${mon_bin}" | tee -a "${log_file}"
        return 1
    fi

    local app_tmp
    app_tmp="$(mktemp)"
    set +e
    (
        cd "${prefix}"
        eval "$(prefix_env_exports "${prefix}")"
        "${app_bin}" --duration-sec "${PROFILE_DURATION_SEC}" \
            "${model_path}" "${input_path}" >"${app_tmp}" 2>&1 &
        local app_pid=$!
        "${mon_bin}" --name "${app_name}" --out "${csv_file}" >>"${log_file}" 2>&1
        local mon_rc=$?
        wait "${app_pid}"
        local app_rc=$?
        echo "--- app ---" >>"${log_file}"
        cat "${app_tmp}" >>"${log_file}"
        rm -f "${app_tmp}"
        echo "app_rc=${app_rc} mon_rc=${mon_rc}" >>"${log_file}"
        [[ "${app_rc}" -eq 0 && "${mon_rc}" -eq 0 ]]
    )
    local rc=$?
    set -e
    echo "exit_code=${rc}" >>"${log_file}"
    return "${rc}"
}

profile_targets() {
    cat <<'EOF'
mobilenetv2_classify_awnn|integration/awnn/mobilenetv2_classify/mobilenetv2.nb|integration/awnn/mobilenetv2_classify/cat.jpg
EOF
}

# 主机编排：tests + 可选 bench/profile
orchestrate() {
    local prefix="$1"
    local artifacts_dir="$2"
    local mode="$3" # local | remote
    local ssh_host="${4:-}"
    local remote_abs="${5:-}"

    mkdir -p "${artifacts_dir}/tests" "${artifacts_dir}/bench" "${artifacts_dir}/profile"
    local orch_log="${artifacts_dir}/orchestrator.log"
    : >"${orch_log}"

    local fail=0
    local passed=0
    local failed=0
    local skipped=0

    echo "=== orchestrate tests (${mode}) ===" | tee -a "${orch_log}"
    echo "prefix(local)=${prefix}" | tee -a "${orch_log}"
    if [[ "${mode}" == "remote" ]]; then
        echo "prefix(remote)=${remote_abs}" | tee -a "${orch_log}"
    fi

    if [[ ! -d "${prefix}/share/percept" ]]; then
        echo "错误: 无效 install（缺少 share/percept）: ${prefix}" | tee -a "${orch_log}"
        return 1
    fi
    if [[ ! -d "${prefix}/tests" ]]; then
        echo "错误: 未找到 ${prefix}/tests" | tee -a "${orch_log}"
        return 1
    fi

    local name rel log_file rc
    while IFS= read -r name; do
        [[ -n "${name}" ]] || continue
        rel="tests/${name}"
        log_file="${artifacts_dir}/tests/${name}.log"
        echo ">>> ${rel}" | tee -a "${orch_log}"
        set +e
        if [[ "${mode}" == "remote" ]]; then
            run_one_remote "${ssh_host}" "${remote_abs}" "${rel}" "${log_file}"
        else
            run_one_local "${prefix}" "${rel}" "${log_file}"
        fi
        rc=$?
        set -e
        if [[ "${rc}" -eq 0 ]]; then
            echo "    PASS" | tee -a "${orch_log}"
            passed=$((passed + 1))
        else
            echo "    FAIL (rc=${rc}) log=${log_file}" | tee -a "${orch_log}"
            failed=$((failed + 1))
            fail=1
        fi
    done < <(discover_executables "${prefix}/tests")

    if [[ "${RUN_BENCH}" -eq 1 ]]; then
        echo "" | tee -a "${orch_log}"
        echo "=== orchestrate benchmarks (loops=${BENCH_LOOPS}) ===" | tee -a "${orch_log}"
        if [[ ! -d "${prefix}/benchmarks" ]]; then
            echo "错误: 未找到 benchmarks/" | tee -a "${orch_log}"
            fail=1
        else
            while IFS= read -r name; do
                [[ -n "${name}" ]] || continue
                rel="benchmarks/${name}"
                log_file="${artifacts_dir}/bench/${name}.log"
                echo ">>> ${rel}" | tee -a "${orch_log}"
                set +e
                if [[ "${mode}" == "remote" ]]; then
                    run_one_remote "${ssh_host}" "${remote_abs}" "${rel}" "${log_file}" "${BENCH_LOOPS}"
                else
                    run_one_local "${prefix}" "${rel}" "${log_file}" "${BENCH_LOOPS}"
                fi
                rc=$?
                set -e
                if [[ "${rc}" -eq 0 ]]; then
                    echo "    PASS" | tee -a "${orch_log}"
                    passed=$((passed + 1))
                else
                    echo "    FAIL (rc=${rc})" | tee -a "${orch_log}"
                    failed=$((failed + 1))
                    fail=1
                fi
            done < <(discover_executables "${prefix}/benchmarks")
        fi
    fi

    if [[ "${RUN_PROFILE}" -eq 1 ]]; then
        echo "" | tee -a "${orch_log}"
        echo "=== orchestrate profiling (duration-sec=${PROFILE_DURATION_SEC}) ===" | tee -a "${orch_log}"
        local app_name model_rel input_rel ran=0
        while IFS='|' read -r app_name model_rel input_rel; do
            [[ -n "${app_name}" ]] || continue
            if [[ ! -x "${prefix}/bin/${app_name}" ]]; then
                echo "skip: ${app_name}（bin 中无此 app）" | tee -a "${orch_log}"
                skipped=$((skipped + 1))
                continue
            fi
            ran=1
            log_file="${artifacts_dir}/profile/${app_name}.log"
            local csv_file="${artifacts_dir}/profile/${app_name}.csv"
            echo ">>> profile ${app_name}" | tee -a "${orch_log}"
            set +e
            if [[ "${mode}" == "remote" ]]; then
                run_one_profile_remote "${ssh_host}" "${remote_abs}" \
                    "${app_name}" "${model_rel}" "${input_rel}" \
                    "${log_file}" "${csv_file}"
            else
                run_one_profile_local "${prefix}" \
                    "${app_name}" "${model_rel}" "${input_rel}" \
                    "${log_file}" "${csv_file}"
            fi
            rc=$?
            set -e
            if [[ "${rc}" -eq 0 ]]; then
                echo "    PASS" | tee -a "${orch_log}"
                passed=$((passed + 1))
            else
                echo "    FAIL (rc=${rc})" | tee -a "${orch_log}"
                failed=$((failed + 1))
                fail=1
            fi
        done < <(profile_targets)
        if [[ "${ran}" -eq 0 ]]; then
            echo "skip: 无可用 profiling 目标" | tee -a "${orch_log}"
        fi
    fi

    {
        echo "passed=${passed}"
        echo "failed=${failed}"
        echo "skipped=${skipped}"
        echo "fail=${fail}"
        echo "finished_at=$(date -Iseconds)"
    } | tee -a "${orch_log}" | tee -a "${artifacts_dir}/meta.txt" >/dev/null

    echo "artifacts: ${artifacts_dir}"
    return "${fail}"
}

run_legacy_on_board() {
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

    "${SSH_WRAP[@]}" "${ssh_host}" "mkdir -p $(printf '%q' "${remote_dir}")" || exit 1
    local remote_abs
    remote_abs="$("${SSH_WRAP[@]}" "${ssh_host}" "cd $(printf '%q' "${remote_dir}") && pwd")" || exit 1
    remote_abs="$(sed -E 's#/+#/#g' <<<"${remote_abs}")"
    remote_abs="${remote_abs%/}"

    scp_install_tree "${install_dir}" "${ssh_host}" "${remote_abs}"

    echo "=== legacy test_runner (remote) ==="
    local remote_cmd="sh $(printf '%q' "${remote_abs}/scripts/test_runner.sh")"
    local arg
    for arg in "${RUNNER_ARGS[@]}"; do
        remote_cmd+=" $(printf '%q' "${arg}")"
    done

    local remote_rc=0
    set +e
    "${SSH_WRAP[@]}" "${ssh_host}" "${remote_cmd}" \
        2>&1 | tee "${artifacts_dir}/orchestrator.log"
    remote_rc=${PIPESTATUS[0]}
    set -e

    mkdir -p "${artifacts_dir}/logs"
    local remote_logs="${remote_abs}/var/percept/logs"
    if "${SSH_WRAP[@]}" "${ssh_host}" "test -d $(printf '%q' "${remote_logs}")"; then
        local -a log_entries=()
        mapfile -t log_entries < <(
            "${SSH_WRAP[@]}" "${ssh_host}" \
                "cd $(printf '%q' "${remote_logs}") && ls -A 2>/dev/null" || true
        )
        local name
        for name in "${log_entries[@]}"; do
            [[ -n "${name}" ]] || continue
            "${SCP_WRAP[@]}" -r \
                "${ssh_host}:${remote_logs}/${name}" \
                "${artifacts_dir}/logs/" 2>/dev/null || true
        done
        echo "pulled: ${remote_logs}/"
    fi

    {
        echo "legacy_runner=1"
        echo "board_spec=${board_spec}"
        echo "remote_abs=${remote_abs}"
        echo "remote_rc=${remote_rc}"
    } >"${artifacts_dir}/meta.txt"
    echo "artifacts: ${artifacts_dir}"
    return "${remote_rc}"
}

run_on_board_orchestrated() {
    local install_dir="$1"
    local board_spec="$2"

    if [[ ! "${board_spec}" =~ ^([^@]+@[^:]+):(.+)$ ]]; then
        echo "错误: --board 格式应为 user@host:remote_path" >&2
        exit 1
    fi
    local ssh_host="${BASH_REMATCH[1]}"
    local remote_dir="${BASH_REMATCH[2]}"

    setup_ssh_transport

    local artifacts_dir="${PROJECT_ROOT}/ci-artifacts/board-logs"
    mkdir -p "${artifacts_dir}"
    rm -rf "${artifacts_dir:?}/"*

    "${SSH_WRAP[@]}" "${ssh_host}" "mkdir -p $(printf '%q' "${remote_dir}")" || {
        echo "错误: 无法创建远程目录: ${remote_dir}" >&2
        exit 1
    }
    local remote_abs
    remote_abs="$("${SSH_WRAP[@]}" "${ssh_host}" "cd $(printf '%q' "${remote_dir}") && pwd")" || {
        echo "错误: 无法访问远程目录: ${remote_dir}" >&2
        exit 1
    }
    remote_abs="$(sed -E 's#/+#/#g' <<<"${remote_abs}")"
    remote_abs="${remote_abs%/}"

    scp_install_tree "${install_dir}" "${ssh_host}" "${remote_abs}"

    {
        echo "board_spec=${board_spec}"
        echo "remote_abs=${remote_abs}"
        echo "legacy_runner=0"
        echo "started_at=$(date -Iseconds)"
    } >"${artifacts_dir}/meta.txt"

    orchestrate "${install_dir}" "${artifacts_dir}" "remote" "${ssh_host}" "${remote_abs}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset) PRESET="$2"; shift 2 ;;
        --board) BOARD="$2"; shift 2 ;;
        --legacy-runner) LEGACY_RUNNER=1; shift ;;
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
    echo "test: preset=${PRESET} mode=board board=${BOARD} legacy=${LEGACY_RUNNER}"
else
    echo "test: preset=${PRESET} mode=host legacy=${LEGACY_RUNNER}"
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
    if [[ "${LEGACY_RUNNER}" -eq 1 ]]; then
        run_legacy_on_board "${INSTALL_DIR}" "${BOARD}"
    else
        run_on_board_orchestrated "${INSTALL_DIR}" "${BOARD}"
    fi
else
    if [[ "${LEGACY_RUNNER}" -eq 1 ]]; then
        run_legacy_runner_local "${INSTALL_DIR}"
    else
        ARTIFACTS="${PROJECT_ROOT}/ci-artifacts/host-logs"
        mkdir -p "${ARTIFACTS}"
        rm -rf "${ARTIFACTS:?}/"*
        orchestrate "${INSTALL_DIR}" "${ARTIFACTS}" "local"
    fi
fi
