#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRESET="x86_64-debug"
INSTALL_DIR=""
BOARD=""
LOOPS=""

PERCEPT_BENCHMARKS_REL="share/percept/benchmarks"

PERCEPT_INSTALL_BENCHMARKS=(
    bench_yolov5_detect_awnn_mapped_vs_hostcopy
)

show_help() {
    echo "deploy_percept 性能 benchmark 脚本（仅运行，不构建、不安装）"
    echo ""
    echo "用法: bash $0 [选项] [-- <bench 参数>]"
    echo ""
    echo "模式 A — build tree（直接运行 build 目录内二进制）:"
    echo "  bash $0 --preset <name> [-- <loops> [model.nb] [input.jpg]]"
    echo ""
    echo "模式 B — install tree（本地 prefix）:"
    echo "  bash $0 --install-dir <path> [-- <loops> [model.nb] [input.jpg]]"
    echo ""
    echo "模式 C — 开发板（install 已 rsync 到板子）:"
    echo "  bash $0 --board <user@host:remote_path> [-- <loops> [model.nb] [input.jpg]]"
    echo ""
    echo "选项:"
    echo "  --preset <name>            CMake preset（build tree 模式）"
    echo "  --install-dir <path>       install prefix 下的 share/percept/benchmarks/"
    echo "  --board <user@host:path>   SSH 到开发板运行 benchmark"
    echo "  --help                     显示帮助"
    echo ""
    echo "构建 benchmark 需: -DENABLE_BENCHMARKS=ON（交叉编译 AWNN preset 示例）"
    echo "  cmake --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release -DENABLE_BENCHMARKS=ON"
    echo ""
    echo "示例:"
    echo "  bash $0 --board orangepi@192.168.1.10:~/deploy_percept -- 50"
    echo "  bash $0 --install-dir install/aarch64-linux-gnu_orange_pi_4_pro_a733 -- 100"
}

collect_bench_args() {
    BENCH_ARGS=()
    if [[ $# -gt 0 && "$1" == "--" ]]; then
        shift
        BENCH_ARGS=("$@")
    fi
}

run_install_tree_benchmarks() {
    local install_dir="$1"
    shift
    collect_bench_args "$@"

    if [[ ! -d "${install_dir}/share/percept" ]]; then
        echo "错误: 无效的 install 目录: ${install_dir}" >&2
        exit 1
    fi
    install_dir="$(cd "${install_dir}" && pwd)"

    local bench_dir="${install_dir}/${PERCEPT_BENCHMARKS_REL}"
    if [[ ! -d "${bench_dir}" ]]; then
        echo "错误: 未找到 benchmark 目录: ${bench_dir}" >&2
        echo "提示: 构建时需 ENABLE_BENCHMARKS=ON 且 INSTALL_BENCHMARKS=ON，并 cmake --install" >&2
        exit 1
    fi

    export PERCEPT_ROOT="${install_dir}/share/percept"
    export LD_LIBRARY_PATH="${install_dir}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    echo "=== install tree benchmarks ==="
    echo "prefix:     ${install_dir}"
    echo "bench dir:  ${bench_dir}"

    local fail=0
    local b
    for b in "${PERCEPT_INSTALL_BENCHMARKS[@]}"; do
        if [[ ! -x "${bench_dir}/${b}" ]]; then
            echo "skip (not installed): ${b}"
            continue
        fi
        echo "=== ${b} ${BENCH_ARGS[*]:-} ==="
        "${bench_dir}/${b}" "${BENCH_ARGS[@]}" || fail=1
    done
    return "${fail}"
}

run_install_tree_benchmarks_remote() {
    local board_spec="$1"
    shift
    collect_bench_args "$@"

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

    local bench_dir="${remote_dir}/${PERCEPT_BENCHMARKS_REL}"
    local args_shell=""
    local a
    for a in "${BENCH_ARGS[@]}"; do
        args_shell+=" $(printf '%q' "${a}")"
    done

    echo "=== install tree benchmarks (remote) ==="
    echo "host:      ${ssh_host}"
    echo "prefix:    ${remote_dir}"
    echo "bench dir: ${bench_dir}"

    local bench_shell=""
    local b
    for b in "${PERCEPT_INSTALL_BENCHMARKS[@]}"; do
        bench_shell+="if [[ -x \"${bench_dir}/${b}\" ]]; then echo '=== ${b} ==='; \"${bench_dir}/${b}\"${args_shell} || FAIL=1; else echo 'skip (not installed): ${b}'; fi;"
    done

    ssh "${ssh_host}" "set -euo pipefail
FAIL=0
export PERCEPT_ROOT=\"${remote_dir}/share/percept\"
export LD_LIBRARY_PATH=\"${remote_dir}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"
if [[ ! -d \"${bench_dir}\" ]]; then
  echo \"错误: 未找到 benchmark 目录: ${bench_dir}\" >&2
  exit 1
fi
${bench_shell}
exit \${FAIL}"
}

run_build_tree_benchmarks() {
    local build_dir="${PROJECT_ROOT}/build/${PRESET}"
    shift
    collect_bench_args "$@"

    if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
        echo "错误: 构建目录未配置: ${build_dir}" >&2
        exit 1
    fi

    export PERCEPT_ROOT="${PROJECT_ROOT}"
    export LD_LIBRARY_PATH="${build_dir}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    echo "=== build tree benchmarks ==="
    echo "preset:    ${PRESET}"
    echo "build dir: ${build_dir}"

    local fail=0
    local b
    for b in "${PERCEPT_INSTALL_BENCHMARKS[@]}"; do
        local exe="${build_dir}/benchmarks/awnn/${b}"
        if [[ ! -x "${exe}" ]]; then
            echo "skip (not built): ${exe}"
            echo "提示: cmake -DENABLE_BENCHMARKS=ON 并 build target ${b}" >&2
            continue
        fi
        echo "=== ${b} ${BENCH_ARGS[*]:-} ==="
        "${exe}" "${BENCH_ARGS[@]}" || fail=1
    done
    return "${fail}"
}

BENCH_TAIL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset) PRESET="$2"; shift 2 ;;
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        --board) BOARD="$2"; shift 2 ;;
        --help) show_help; exit 0 ;;
        --)
            BENCH_TAIL=("$@")
            break
            ;;
        *)
            echo "错误: 未知参数: $1" >&2
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
    run_install_tree_benchmarks_remote "${BOARD}" "${BENCH_TAIL[@]}"
    exit $?
fi

if [[ -n "${INSTALL_DIR}" ]]; then
    if [[ "${INSTALL_DIR}" != /* ]]; then
        INSTALL_DIR="${PROJECT_ROOT}/${INSTALL_DIR}"
    fi
    run_install_tree_benchmarks "${INSTALL_DIR}" "${BENCH_TAIL[@]}"
    exit $?
fi

run_build_tree_benchmarks "${BENCH_TAIL[@]}"
exit $?
