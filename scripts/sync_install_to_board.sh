#!/bin/bash
# 将本地 install 树复制到开发板（仅 scp，不 build、不跑测）
# 默认 quick：只传脚本内清单；--full：清空远端后整树上传。
# 测试请用 scripts/test.sh；CI/CD 与首次上板请加 --full。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 远端目录名必须为此（降低误清风险）
REMOTE_BASENAME_REQUIRED="deploy_percept"

# 默认 quick 清单（相对 --install-dir；当前主测板 = Sigmastar）
# 其它平台 fixture / unit 数据请用 --full；后续合并板型时再扩展本数组。
QUICK_PATHS=(
    benchmarks/
    bin/
    include/
    lib/
    # lib/libdeploy_percept_core.a
    # lib/libdeploy_percept_utils.a
    profiling/
    scripts/
    share/
    tests/
)

INSTALL_DIR=""
BOARD=""
FULL=0

SSH_WRAP=()
SCP_WRAP=()
SSH_HOST=""
REMOTE_ABS=""

show_help() {
    echo "deploy_percept 板端复制（仅 scp install → 开发板）"
    echo ""
    echo "用法: bash $0 --install-dir <dir> --board <user@host:/abs/.../deploy_percept> [--full]"
    echo ""
    echo "选项:"
    echo "  --install-dir <dir>   本地 install 根目录（必填）"
    echo "  --board <spec>        user@host:绝对路径（必填；basename 须为 ${REMOTE_BASENAME_REQUIRED}）"
    echo "  --full                全量：清空远端目录后整树 scp（默认否，走 QUICK_PATHS）"
    echo "  --help                显示帮助"
    echo ""
    echo "默认 quick 清单（写在本脚本 QUICK_PATHS）:"
    local p
    for p in "${QUICK_PATHS[@]}"; do
        echo "  - ${p}"
    done
    echo ""
    echo "认证: export SSHPASS='...'（需主机安装 sshpass；勿提交仓库）"
    echo "新板 / CI 请加 --full；日常迭代用默认 quick。"
    echo ""
    echo "示例:"
    echo "  export SSHPASS='...'"
    echo "  bash $0 --install-dir install/armv7l-linux-gnueabihf_comake_pi_d1_ssd2355 \\"
    echo "    --board root@192.168.1.102:/customer/tmp/deploy_percept"
    echo "  bash $0 --install-dir install/... --board root@host:/customer/tmp/deploy_percept --full"
}

setup_ssh_transport() {
    if [[ -z "${SSHPASS:-}" ]]; then
        echo "错误: 需要设置 SSHPASS" >&2
        echo "示例: export SSHPASS='your_password'" >&2
        return 1
    fi
    if ! command -v sshpass >/dev/null 2>&1; then
        echo "错误: 未找到 sshpass（如: sudo apt install sshpass）" >&2
        return 1
    fi
    SSH_WRAP=(sshpass -e ssh -n -o StrictHostKeyChecking=accept-new)
    SCP_WRAP=(sshpass -e scp -o StrictHostKeyChecking=accept-new)
}

# 解析并校验 --board → SSH_HOST + REMOTE_ABS
parse_board_spec() {
    local board_spec="$1"
    if [[ ! "${board_spec}" =~ ^([^@]+@[^:]+):(.+)$ ]]; then
        echo "错误: --board 格式应为 user@host:/absolute/path/${REMOTE_BASENAME_REQUIRED}" >&2
        return 1
    fi
    SSH_HOST="${BASH_REMATCH[1]}"
    REMOTE_ABS="${BASH_REMATCH[2]}"
    REMOTE_ABS="${REMOTE_ABS%/}"

    if [[ "${REMOTE_ABS}" != /* ]]; then
        echo "错误: 远端路径须为绝对路径（以 / 开头），不支持 ~ 或相对路径: ${REMOTE_ABS}" >&2
        return 1
    fi
    if [[ "$(basename "${REMOTE_ABS}")" != "${REMOTE_BASENAME_REQUIRED}" ]]; then
        echo "错误: 远端路径 basename 必须是 '${REMOTE_BASENAME_REQUIRED}'，当前: $(basename "${REMOTE_ABS}")" >&2
        return 1
    fi

    case "${REMOTE_ABS}" in
        /|"/root"|"/home"|"/customer"|"/tmp"|"/var"|"/mnt"|"/userdata")
            echo "错误: 拒绝危险远端路径: ${REMOTE_ABS}" >&2
            return 1
            ;;
    esac

    local depth
    depth="$(awk -F'/' '{print NF-1}' <<<"${REMOTE_ABS}")"
    if [[ "${depth}" -lt 2 ]]; then
        echo "错误: 远端路径过浅（至少两级，如 /customer/deploy_percept）: ${REMOTE_ABS}" >&2
        return 1
    fi
}

remote_clear_dir() {
    local remote_abs="$1"
    "${SSH_WRAP[@]}" "${SSH_HOST}" \
        "mkdir -p $(printf '%q' "${remote_abs}") && cd $(printf '%q' "${remote_abs}") && rm -rf ./* ./.[!.]* ./..?* 2>/dev/null || true"
}

# 删除远端单个相对路径后 scp 对应本地条目
scp_one_rel() {
    local install_dir="$1"
    local rel="$2"
    local local_path="${install_dir}/${rel}"
    # 去掉尾部 /
    local_path="${local_path%/}"
    local rel_clean="${rel%/}"

    if [[ ! -e "${local_path}" ]]; then
        echo "警告: 清单项不存在，跳过: ${local_path}" >&2
        return 0
    fi

    local remote_path="${REMOTE_ABS}/${rel_clean}"
    local remote_parent
    remote_parent="$(dirname "${remote_path}")"

    echo "=== sync ${rel_clean} → ${SSH_HOST}:${remote_path} ===" >&2
    "${SSH_WRAP[@]}" "${SSH_HOST}" \
        "mkdir -p $(printf '%q' "${remote_parent}") && rm -rf $(printf '%q' "${remote_path}")"

    if [[ -d "${local_path}" ]]; then
        "${SCP_WRAP[@]}" -r "${local_path}" "${SSH_HOST}:${remote_parent}/"
    else
        "${SCP_WRAP[@]}" "${local_path}" "${SSH_HOST}:${remote_path}"
    fi
}

sync_full() {
    local install_dir="$1"
    local -a entries=()

    echo "=== clear remote ${SSH_HOST}:${REMOTE_ABS}/ (--full) ===" >&2
    remote_clear_dir "${REMOTE_ABS}"

    echo "=== scp full tree → ${SSH_HOST}:${REMOTE_ABS}/ ===" >&2
    shopt -s nullglob dotglob
    entries=("${install_dir}"/*)
    shopt -u nullglob dotglob
    if [[ ${#entries[@]} -eq 0 ]]; then
        echo "错误: install 目录为空: ${install_dir}" >&2
        return 1
    fi
    "${SCP_WRAP[@]}" -r "${entries[@]}" "${SSH_HOST}:${REMOTE_ABS}/"
}

sync_quick() {
    local install_dir="$1"
    local rel

    echo "=== quick sync → ${SSH_HOST}:${REMOTE_ABS}/ ===" >&2
    "${SSH_WRAP[@]}" "${SSH_HOST}" "mkdir -p $(printf '%q' "${REMOTE_ABS}")"

    if [[ ${#QUICK_PATHS[@]} -eq 0 ]]; then
        echo "错误: QUICK_PATHS 为空" >&2
        return 1
    fi
    for rel in "${QUICK_PATHS[@]}"; do
        [[ -n "${rel}" ]] || continue
        scp_one_rel "${install_dir}" "${rel}"
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        --board) BOARD="$2"; shift 2 ;;
        --full) FULL=1; shift ;;
        --help) show_help; exit 0 ;;
        *)
            echo "错误: 未知参数: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

if [[ -z "${INSTALL_DIR}" || -z "${BOARD}" ]]; then
    echo "错误: 必须指定 --install-dir 与 --board" >&2
    show_help
    exit 1
fi

if [[ ! -d "${INSTALL_DIR}" ]]; then
    echo "错误: --install-dir 不是目录: ${INSTALL_DIR}" >&2
    exit 1
fi
# 规范化为绝对路径，便于日志
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"

# 先校验路径规则，再要密码/ssh（错误更直观）
parse_board_spec "${BOARD}"
setup_ssh_transport

echo "sync: install_dir=${INSTALL_DIR} board=${SSH_HOST}:${REMOTE_ABS} full=${FULL}" >&2

"${SSH_WRAP[@]}" "${SSH_HOST}" "mkdir -p $(printf '%q' "${REMOTE_ABS}")" || {
    echo "错误: 无法创建远程目录: ${REMOTE_ABS}" >&2
    exit 1
}

if [[ "${FULL}" -eq 1 ]]; then
    sync_full "${INSTALL_DIR}"
else
    sync_quick "${INSTALL_DIR}"
fi

echo "synced: ${SSH_HOST}:${REMOTE_ABS} (full=${FULL})" >&2
