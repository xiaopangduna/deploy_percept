# benchmarks/sigmastar

Sigmastar IPU 性能 benchmark。与 `integration/sigmastar`（正确性）和 `apps/*_sigmastar`（演示）互补。

## bench_ipu_img

模型无关压测：只依赖 `.img` + IPU firmware，随机填充输入，不做业务 pre/post。  
按模型 `BatchMode` 走 `MI_IPU_Invoke2`（N_BUF）或 `MI_IPU_Invoke2Custom`（ONE_BUF）。  
每个模型在压测前打印 SDK 风格详情（Variable/Work/Batch、Input/Output 的 name/dtype/shape/size 等）。

**渊源**：IPU 调用路径源自 SigmaStar SDK `dla_show_img_info`（offline info / BatchMode / 张量缓冲 / Invoke2·Invoke2Custom）；指标与跑法对齐 OpenDLA `prog_benchmark`，并增加 warmup、多轮统计与输入填充。无参自动扫模型目录时走 `test_common::fixture_dir`（与集成测同一套 fixture 根路径）。

### 构建

```bash
bash scripts/build.sh \
  --preset armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release \
  --install \
  -DENABLE_BENCHMARKS=ON
```

需 `SIGMASTAR_FOUND`（`third_party/.../sigmastar/release`）。产物：`benchmarks/bench_ipu_img`。

### 用法

```bash
cd /customer/tmp/deploy_percept   # install prefix
export LD_LIBRARY_PATH=$PWD/lib PERCEPT_ROOT=$PWD/share/percept

# 自动：扫 share/percept/fixtures/integration/sigmastar/*.img
./benchmarks/bench_ipu_img

# 单模型 / 目录
./benchmarks/bench_ipu_img ./share/percept/fixtures/integration/sigmastar/mobilenet_v2_224_ssd2355.img
./benchmarks/bench_ipu_img /mnt/SModel/my_models/ --batch 1
```

| 选项 | 默认 | 说明 |
|------|------|------|
| `--batch N` | 1 | N_BUF 模式 batch；ONE_BUF 用模型内嵌值 |

固定参数：`warmup=5`，`loops=50`，`firmware=/config/dla/ipu_firmware.bin`，`fill=random`。

### 输出

- **单模型详情**：Variable size、Work/Batch mode、各 Input/Output 的 name / dtype / shape / size（及 NCHW / INT16 quant）
- **汇总表**：avg / p50 / p99 / min、bw、ROM、RAM

### 指标

| 列 | 含义 |
|----|------|
| avg/p50/p99/min (ms) | 墙钟 invoke 时延（仅包 `Invoke2*`） |
| bw(KB) | `batch * RuntimeInfo.u64BandWidth / 1024` 的均值 |
| ROM(KB) | `.img` 文件大小 |
| RAM(KB) | `(ROM + VariableBuffer + batch * output_buf) / 1024` 估算 |

### 与其它工具

| 工具 | 用途 |
|------|------|
| `bench_ipu_img` | IPU 侧模型吞吐 / 带宽 / 内存 + 模型 IO 详情 |
| `apps/*_sigmastar` | 端到端演示 |
| `tests/test_*_sigmastar` | 正确性 |
| `profiling/percept_monitor` | 可选旁路监控本进程 CPU/RSS |
