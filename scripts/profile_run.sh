#!/bin/sh
# 无参资源画像：在 install 根目录先启被测 app，再挂 percept_monitor。
# 用法: cd /path/to/deploy_percept && sh scripts/profile_run.sh
set -eu

if [ "$#" -ne 0 ]; then
    echo "用法: sh $0" >&2
    echo "  在 deploy_percept install 根目录执行，无参数。" >&2
    exit 1
fi

if [ ! -x ./bin/yolov8_detect_sigmastar ] || [ ! -x ./profiling/percept_monitor ]; then
    echo "错误: 请在 install 根目录执行（需可执行的 bin/ 与 profiling/）" >&2
    exit 1
fi

OUT="./var/profile/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}"

APP_PID=""
cleanup() {
    if [ -n "${APP_PID}" ] && kill -0 "${APP_PID}" 2>/dev/null; then
        kill "${APP_PID}" 2>/dev/null || true
        wait "${APP_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "=== profile_run out=${OUT} ==="

# --- group 1: sigmastar yolov8 detect ---
# app 90s；monitor 预热 10s（summary 不含），稳态约 60s+；其余用默认
./bin/yolov8_detect_sigmastar --duration-sec 90 \
    share/percept/fixtures/integration/sigmastar/yolov8n_640x640_ssd2355.img \
    share/percept/fixtures/integration/sigmastar/dog.jpg \
    >"${OUT}/app.log" 2>&1 &
APP_PID=$!

set +e
./profiling/percept_monitor --name yolov8_detect_sigmastar --warmup-sec 10 \
    >"${OUT}/monitor.log" 2>&1
MON_RC=$?
wait "${APP_PID}"
APP_RC=$?
set -e
APP_PID=""

echo "--- app.log (tail) ---"
tail -n 20 "${OUT}/app.log" 2>/dev/null || true
echo "--- monitor summary ---"
if grep -q '=== percept_monitor summary ===' "${OUT}/monitor.log" 2>/dev/null; then
    sed -n '/=== percept_monitor summary ===/,$p' "${OUT}/monitor.log"
else
    tail -n 30 "${OUT}/monitor.log" 2>/dev/null || true
fi
echo "logs: ${OUT}"

if [ "${APP_RC}" -ne 0 ] || [ "${MON_RC}" -ne 0 ]; then
    echo "FAIL app_rc=${APP_RC} monitor_rc=${MON_RC}" >&2
    exit 1
fi
echo "PASS"
exit 0
