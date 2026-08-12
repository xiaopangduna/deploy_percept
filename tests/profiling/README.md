# profiling

旁路资源画像工具。与 `tests/benchmarks/`（单进程延迟）互补：被测程序在 `apps/`，本目录只提供 monitor。

## percept_monitor

按**进程名**定位目标（匹配 `/proc/<pid>/cmdline` argv[0] 或 `exe` 的 basename），采样至进程退出。

| 指标 | 来源 |
|------|------|
| 进程 CPU% | `/proc/<pid>/stat` utime+stime |
| RSS / max / HWM | `/proc/<pid>/status` |
| Threads | `/proc/<pid>/status` |
| NPU load% | `/sys/kernel/debug/viplite/core_loading`（约 30s 历史窗口；debugfs 需 root） |
| IPU util% | `/proc/mi_modules/mi_ipu/debug_hal/ipu_status`（累计忙时 µs 差分；Sigmastar） |

| 参数 | 默认 | 说明 |
|------|------|------|
| `--name` | 必填 | 进程名（可执行文件 basename） |
| `--interval-ms` | 100 | 采样间隔 |
| `--warmup-sec` | 5 | 前 T 秒不计入 summary（建议 5–10） |
| `--wait-sec` | 60 | 等待进程出现的超时 |
| `--out` | 无 | 可选 CSV |

### NPU（VIP / AWNN）

`core_loading` 示例：

```text
VIP Total Time=30009 ms
History Loading -----> Core0: Inference Time=24063 ms (80%)
```

解析括号内百分比为 `npu_load_percent`。无节点/无权限时跳过 NPU，CPU/RSS 仍正常。

### IPU（Sigmastar）

对齐 SDK `ipu_utilization`：读 `ipu_status` 累计忙时（µs），用相邻两拍墙钟间隔差分：

```text
ipu_util% = 100 * Δbusy_us / (Δwall_sec * 1e6)
```

仅 core0。无节点（如 AWNN 板）则跳过 IPU，不影响其它指标。短 `--interval-ms` 时单点可能较噪，summary 看 warmup 后 avg/max。

与独立工具差异：`ipu_utilization -t 3` 常驻只打利用率；`percept_monitor` 旁路挂目标进程，随进程退出，并同时采 CPU/RSS 等。

## 板端用 root（读 NPU）

`/sys/kernel/debug/viplite/` 在 debugfs 上，普通用户无权访问。专用开发板 / CI **直接 root SSH**，无需 sudoers：

```bash
# BOARD_SPEC / --board 示例（绝对路径）
root@192.168.1.101:/root/deploy_percept

export SSHPASS='...'
bash scripts/test.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --board root@192.168.1.101:/root/deploy_percept --full
```

root 下 `percept_monitor` 可直接读 `core_loading`。`SSHPASS` 只用于 SSH 登录。`test.sh` 只编排正确性测试；资源画像用下面的 `profile_run.sh`。

## 板端一键画像（推荐）

在 install 根目录无参执行（命令写死在脚本内：先 app 后 monitor）：

```bash
cd /customer/tmp/deploy_percept
sh scripts/profile_run.sh
```

主机编排也可：`bash scripts/test.sh --preset … --board … --profile`（在 tests/、可选 benchmarks/ 之后跑 `profile_run.sh`，日志 `ci-artifacts/board-logs/profile_run.log`）。

当前第一组：

1. `./bin/yolov8_detect_sigmastar --duration-sec 90` + sigmastar fixture  
2. `./profiling/percept_monitor --name yolov8_detect_sigmastar --warmup-sec 10`（interval/wait 用默认）

时间关系：app 墙钟 90s；monitor 前 10s 为预热（不进 summary），其后至进程退出为稳态采样（约 60s+）。monitor 本身跟目标进程退出，无单独「监控时长」参数。

日志：`var/profile/<时间戳>/{app.log,monitor.log}`。

## 板端手跑示例

### AWNN + mobilenetv2

```bash
cd /root/deploy_percept   # install prefix（root 会话）

./bin/mobilenetv2_classify_awnn --duration-sec 60 \
  share/percept/fixtures/integration/awnn/mobilenetv2_a733.nb \
  share/percept/fixtures/integration/awnn/cat.jpg &

./profiling/percept_monitor --name mobilenetv2_classify_awnn
```

### Sigmastar + yolov8 detect

```bash
cd /customer/tmp/deploy_percept

./bin/yolov8_detect_sigmastar --duration-sec 60 \
  share/percept/fixtures/integration/sigmastar/yolov8n_640x640_ssd2355.img \
  share/percept/fixtures/integration/sigmastar/dog.jpg &

./profiling/percept_monitor --name yolov8_detect_sigmastar
```

有 `ipu_status` 时启动会打印 `ipu: enabled (ipu_status)`，summary 含 `ipu_util_avg` / `ipu_util_max`。

app `--duration-sec T`：是 **app 循环墙钟时间**（默认推荐 **60**，配合 core_loading ~30s 窗口）。monitor 跟随目标进程退出。延迟分段请用 `benchmarks/bench_*`。

## 日志与回传

`test.sh --board` 写入源码树：

```text
ci-artifacts/board-logs/   # 或 host-logs/
  meta.txt
  test_run.log
```

Board CI 上传 artifact `board-logs-<sha>`。
