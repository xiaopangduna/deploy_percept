# profiling

旁路资源画像工具。与 `tests/benchmarks/`（单进程延迟）互补：被测程序在 `apps/`，本目录只提供 monitor。

## percept_monitor

按**进程名**定位目标（匹配 `/proc/<pid>/cmdline` argv[0] 或 `exe` 的 basename），采样至进程退出。

| 指标 | 来源 |
|------|------|
| 进程 CPU% | `/proc/<pid>/stat` utime+stime |
| RSS / max / HWM | `/proc/<pid>/status` |
| Threads | `/proc/<pid>/status` |

| 参数 | 默认 | 说明 |
|------|------|------|
| `--name` | 必填 | 进程名（可执行文件 basename） |
| `--interval-ms` | 100 | 采样间隔 |
| `--warmup-sec` | 5 | 前 T 秒不计入 summary（建议 5–10） |
| `--wait-sec` | 60 | 等待进程出现的超时 |
| `--out` | 无 | 可选 CSV |

## 板端示例（与 mobilenetv2 demo）

```bash
cd ~/deploy_percept   # install prefix
export LD_LIBRARY_PATH=$PWD/lib

./bin/mobilenetv2_classify_awnn --duration-sec 30 \
  share/percept/fixtures/integration/awnn/mobilenetv2_classify/mobilenetv2.nb \
  share/percept/fixtures/integration/awnn/mobilenetv2_classify/cat.jpg &

./profiling/percept_monitor --name mobilenetv2_classify_awnn
```

也可先启动 monitor（`--wait-sec` 内等待 app 出现），再启动 app。

`--profile T` / app `--duration-sec T`：都是 **app 循环墙钟时间**；monitor 跟随目标进程退出。延迟分段请用 `benchmarks/bench_*`。

## 经 test_runner / test.sh

```bash
bash scripts/test.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --board orangepi@<ip>:~/deploy_percept --profile 30

./scripts/test_runner.sh --profile 30
```

当前内置目标：`mobilenetv2_classify_awnn`（bin 中不存在则 skip）。
