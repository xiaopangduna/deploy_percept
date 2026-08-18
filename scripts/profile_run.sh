#!/bin/sh
# 无参资源画像：在 install 根目录按列表依次「先 app 后 monitor」。
# 用法: cd /path/to/deploy_percept && sh scripts/profile_run.sh
#
# 按 ./bin/*_<后缀> 是否存在选平台场景（*_sigmastar / *_awnn），命令与 fixture 仍写死。
# 新增一组：在对应后缀块里再调一次 run_one 即可。
set -eu

if [ "$#" -ne 0 ]; then
    echo "用法: sh $0" >&2
    echo "  在 deploy_percept install 根目录执行，无参数。" >&2
    exit 1
fi

if [ ! -x ./profiling/percept_monitor ]; then
    echo "错误: 请在 install 根目录执行（需 ./profiling/percept_monitor）" >&2
    exit 1
fi

OUT="./var/profile/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}"

APP_PID=""
FAIL=0

cleanup() {
    if [ -n "${APP_PID}" ] && kill -0 "${APP_PID}" 2>/dev/null; then
        kill "${APP_PID}" 2>/dev/null || true
        wait "${APP_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# run_one <monitor_name> <log_tag> <app> [app_args...]
# 先后台启 app，再挂 percept_monitor（warmup=10，其余默认），再 wait。
run_one() {
    mon_name=$1
    tag=$2
    shift 2

    if [ ! -x "$1" ]; then
        echo "SKIP [${tag}]: 无可执行文件 $1" >&2
        FAIL=1
        return 0
    fi

    echo "----------------"
    echo "[${tag}] app: $*"
    echo "[${tag}] monitor: --name ${mon_name} --warmup-sec 10"

    "$@" >"${OUT}/app_${tag}.log" 2>&1 &
    APP_PID=$!

    set +e
    ./profiling/percept_monitor --name "${mon_name}" --warmup-sec 10 \
        >"${OUT}/monitor_${tag}.log" 2>&1
    mon_rc=$?
    wait "${APP_PID}"
    app_rc=$?
    set -e
    APP_PID=""

    echo "--- app_${tag}.log (tail) ---"
    tail -n 15 "${OUT}/app_${tag}.log" 2>/dev/null || true
    echo "--- monitor_${tag} summary ---"
    if grep -q '=== percept_monitor summary ===' "${OUT}/monitor_${tag}.log" 2>/dev/null; then
        sed -n '/=== percept_monitor summary ===/,$p' "${OUT}/monitor_${tag}.log"
    else
        tail -n 20 "${OUT}/monitor_${tag}.log" 2>/dev/null || true
    fi

    if [ "${app_rc}" -ne 0 ] || [ "${mon_rc}" -ne 0 ]; then
        echo "FAIL [${tag}] app_rc=${app_rc} monitor_rc=${mon_rc}" >&2
        FAIL=1
    else
        echo "PASS [${tag}]"
    fi
    echo "################"
}

# 无匹配 glob 时 sh 保留字面量 ./bin/*_awnn，-x 失败 → 视为无此后缀。
has_bin_suffix() {
    suf=$1
    for f in ./bin/*_"${suf}"; do
        if [ -x "$f" ]; then
            return 0
        fi
    done
    return 1
}

echo "=== profile_run out=${OUT} ==="

# ========== 场景列表（按 bin 后缀选块；块内依次执行）==========
# app 90s / 组；monitor 预热 10s（summary 不含），稳态约 60s+
# Sigmastar / AWNN 均为三组，约 270s+

RAN_PLATFORM=0

if has_bin_suffix sigmastar; then
    RAN_PLATFORM=1
    echo "=== platform: sigmastar ==="

    run_one yolov8_detect_sigmastar yolov8_detect \
        ./bin/yolov8_detect_sigmastar --duration-sec 90 \
        share/percept/fixtures/integration/sigmastar/yolov8n_640x640_ssd2355.img \
        share/percept/fixtures/integration/sigmastar/dog.jpg

    run_one mobilenetv2_classify_sigmastar mobilenetv2_classify \
        ./bin/mobilenetv2_classify_sigmastar --duration-sec 90 \
        share/percept/fixtures/integration/sigmastar/mobilenet_v2_224_ssd2355.img \
        share/percept/fixtures/integration/sigmastar/apple.jpg

    run_one yolo_cls_sigmastar yolo_cls \
        ./bin/yolo_cls_sigmastar --duration-sec 90 \
        share/percept/fixtures/integration/sigmastar/yolov8n_640x640_ssd2355.img \
        share/percept/fixtures/integration/sigmastar/mobilenet_v2_224_ssd2355.img \
        share/percept/fixtures/integration/sigmastar/dog.jpg \
        share/percept/fixtures/integration/sigmastar/apple.jpg

    # 示例：再加一组时复制上面，改 name / tag / 命令即可（app 需支持 --duration-sec）
    # run_one yolov8_seg_sigmastar yolov8_seg \
    #     ./bin/yolov8_seg_sigmastar --duration-sec 90 \
    #     share/percept/fixtures/integration/sigmastar/yolov8n_seg_640x640_ssd2355.img \
    #     share/percept/fixtures/integration/sigmastar/bus.jpg
fi

if has_bin_suffix awnn; then
    RAN_PLATFORM=1
    echo "=== platform: awnn ==="

    run_one yolov8_detect_awnn yolov8_detect_awnn \
        ./bin/yolov8_detect_awnn --duration-sec 90 \
        share/percept/fixtures/integration/awnn/yolov8_a733.nb \
        share/percept/fixtures/integration/awnn/dog.jpg

    run_one mobilenetv2_classify_awnn mobilenetv2_classify_awnn \
        ./bin/mobilenetv2_classify_awnn --duration-sec 90 \
        share/percept/fixtures/integration/awnn/mobilenetv2_a733.nb \
        share/percept/fixtures/integration/awnn/cat.jpg

    run_one yolo_cls_awnn yolo_cls_awnn \
        ./bin/yolo_cls_awnn --duration-sec 90 \
        share/percept/fixtures/integration/awnn/yolov8_a733.nb \
        share/percept/fixtures/integration/awnn/mobilenetv2_a733.nb \
        share/percept/fixtures/integration/awnn/dog.jpg \
        share/percept/fixtures/integration/awnn/cat.jpg
fi

if [ "${RAN_PLATFORM}" -eq 0 ]; then
    echo "错误: ./bin 下没有 *_sigmastar / *_awnn 可执行文件" >&2
    FAIL=1
fi

# ==================================================================

echo "logs: ${OUT}"
if [ "${FAIL}" -ne 0 ]; then
    echo "FAIL" >&2
    exit 1
fi
echo "PASS"
exit 0
