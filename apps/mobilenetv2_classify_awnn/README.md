# mobilenetv2_classify_awnn

全志 VIPLite MobileNetV2 ImageNet 分类（Orange Pi 4 Pro A733）。推理经 `deploy_percept::engine::AwnnEngine`，后处理见 `ClassificationPostProcessAwnn`（与 `tmp/awnpu_model_zoo-.../examples/mobilenetv2` 单路 FP32 logits + Top-K 一致）。

## 资源（install → share/percept/fixtures/integration/awnn/）

| 文件 | 说明 |
|------|------|
| `mobilenetv2_a733.nb` | A733 Acuity 导出模型 |
| `cat.jpg` | 默认输入图（任意尺寸，demo 内 resize 至 224×224） |

模型与图片位于 `tests/fixtures/integration/awnn/`，install 时复制到 `share/percept/fixtures/`。

## 模型从哪来

训练 → ONNX → Acuity → `.nb` 的全流程入口见：

→ [src/train/exports/awnn/mobilenet_v2/README.md](../../../train/exports/awnn/mobilenet_v2/README.md)

转换完成后将产物复制为 fixture（路径相对 `deploy_percept/`）：

```bash
cp ../train/exports/awnn/mobilenet_v2/model/mobilenetv2-12_pcq_a733.nb \
  tests/fixtures/integration/awnn/mobilenetv2_a733.nb
```

要求：`ADD_PREPROC_NODE=True`、`PREPROC_TYPE=IMAGE_RGB`、`ADD_POSTPROC_NODE=True`（NPU 内 mean/scale + FP32 输出）。

## 构建与安装

```bash
cmake --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release
cmake --build --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --target mobilenetv2_classify_awnn bench_mobilenetv2_classify
bash scripts/build.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release --install
```

## 运行

```bash
cd ~/deploy_percept
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./bin/mobilenetv2_classify_awnn
```

显式指定路径：

```bash
./bin/mobilenetv2_classify_awnn \
  share/percept/fixtures/integration/awnn/mobilenetv2_a733.nb \
  share/percept/fixtures/integration/awnn/cat.jpg
```

导出 unit 单测 fixture（板端，后处理 golden）：

```bash
./bin/mobilenetv2_classify_awnn \
  share/percept/fixtures/integration/awnn/mobilenetv2_a733.nb \
  share/percept/fixtures/integration/awnn/cat.jpg \
  /tmp/model_outputs.npz
```

拷回 PC 后置于 `tests/fixtures/unit/post_process/mobilenetv2_classify_awnn/model_outputs.npz`。

### 资源画像（`--duration-sec`）

按墙钟循环推理，供 `profiling/percept_monitor` 旁路采样。`--profile T` / `--duration-sec T` 都是 **app 循环持续时间**（推荐 ≥60）；monitor 跟进程退出。读 NPU 请用 **root** 会话（见 profiling README）。

```bash
./bin/mobilenetv2_classify_awnn --duration-sec 60 \
  share/percept/fixtures/integration/awnn/mobilenetv2_a733.nb \
  share/percept/fixtures/integration/awnn/cat.jpg &

./profiling/percept_monitor --name mobilenetv2_classify_awnn
```

详见 [tests/profiling/README.md](../../tests/profiling/README.md)（手跑 app + `percept_monitor`；`test.sh` 不再带 `--profile`）。

## 预处理

1. 读取 BGR 原图
2. resize 到模型输入 224×224
3. BGR → RGB，**HWC UINT8** 写入 VIP input buffer（mean/scale 由 NPU 预处理节点完成）

## 模型 I/O

### 输入

| 项 | 值 |
|----|-----|
| dtype | UINT8 |
| VIP `input_sizes` | `[C, H, W, N] = [3, 224, 224, 1]` |
| buffer 布局 | RGB **HWC** 交错 |
| 字节数 | 150 528（224×224×3） |

### 输出（1 路 FP32）

| 项 | 值 |
|----|-----|
| shape | `[1, 1000]` logits |
| 后处理 | Top-K + ImageNet 标签 |

## 正确性回归

```bash
bash scripts/test.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --board root@<ip>:/root/deploy_percept --full
```

对应 `tests/test_mobilenetv2_classify_awnn`。Top-5 class_id 真值：`{281, 282, 285, 287, 478}`。

## 性能 benchmark

```bash
# install prefix 内直接跑 bench（test.sh 不再带 --bench）
export PERCEPT_ROOT=$PWD/share/percept
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./benchmarks/bench_mobilenetv2_classify 50
```
