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

| 参数 | 默认 | 说明 |
|------|------|------|
| `--name` | 必填 | 进程名（可执行文件 basename） |
| `--interval-ms` | 100 | 采样间隔 |
| `--warmup-sec` | 5 | 前 T 秒不计入 summary（建议 5–10） |
| `--wait-sec` | 60 | 等待进程出现的超时 |
| `--out` | 无 | 可选 CSV |

`core_loading` 示例：

```text
VIP Total Time=30009 ms
History Loading -----> Core0: Inference Time=24063 ms (80%)
```

解析括号内百分比为 `npu_load_percent`。无节点/无权限时跳过 NPU，CPU/RSS 仍正常。

## 板端用 root（读 NPU）

`/sys/kernel/debug/viplite/` 在 debugfs 上，普通用户无权访问。专用开发板 / CI **直接 root SSH**，无需 sudoers：

```bash
# BOARD_SPEC / --board 示例
root@192.168.1.101:~/deploy_percept

export SSHPASS='...'
bash scripts/test.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --board root@192.168.1.101:~/deploy_percept --profile 60
```

root 下 `percept_monitor` 可直接读 `core_loading`。`SSHPASS` 只用于 SSH 登录。

## 板端示例（与 mobilenetv2 demo）

```bash
cd ~/deploy_percept   # install prefix（root 会话）
export LD_LIBRARY_PATH=$PWD/lib

./bin/mobilenetv2_classify_awnn --duration-sec 60 \
  share/percept/fixtures/integration/awnn/mobilenetv2_classify/mobilenetv2.nb \
  share/percept/fixtures/integration/awnn/mobilenetv2_classify/cat.jpg &

./profiling/percept_monitor --name mobilenetv2_classify_awnn
```

`--profile T` / app `--duration-sec T`：都是 **app 循环墙钟时间**（默认推荐 **60**，配合 core_loading ~30s 窗口）。monitor 跟随目标进程退出。延迟分段请用 `benchmarks/bench_*`。

## 经 test_runner / test.sh

```bash
bash scripts/test.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --board root@<ip>:~/deploy_percept --profile 60

./scripts/test_runner.sh --profile 60
```

当前内置目标：`mobilenetv2_classify_awnn`（bin 中不存在则 skip）。

## 日志与回传

`test_runner` 写入 install 树：

```text
var/percept/logs/
  meta.txt
  test.log       # 全部正确性测试（段分隔）
  bench.log      # 全部 latency bench
  profile.log    # monitor + app 文本
  profile.csv    # monitor 时间序列（含 npu_load）
```

`--board` 模式结束后，`test.sh` 拉到源码树 `ci-artifacts/board-logs/`；Board CI 上传 artifact `board-logs-<sha>`。
