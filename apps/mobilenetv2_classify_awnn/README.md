# mobilenetv2_classify_awnn

全志 VIPLite MobileNetV2 ImageNet 分类（Orange Pi 4 Pro A733）。推理经 `deploy_percept::engine::AwnnEngine`，后处理见 `ClassificationPostProcessAwnn`（与 `tmp/awnpu_model_zoo-.../examples/mobilenetv2` 单路 FP32 logits + Top-K 一致）。

## 资源（install → share/percept/fixtures/integration/awnn/mobilenetv2_classify/）

| 文件 | 说明 |
|------|------|
| `mobilenetv2.nb` | A733 Acuity 导出模型 |
| `cat.jpg` | 默认输入图（任意尺寸，demo 内 resize 至 224×224） |

模型与图片位于 `tests/fixtures/integration/awnn/mobilenetv2_classify/`，install 时复制到 `share/percept/fixtures/`。

## 模型转换

参考 `tmp/awnpu_model_zoo-v1.0.0-20260423-f562dd16/examples/mobilenetv2/`：

1. 下载 [mobilenetv2-12.onnx](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet)
2. Acuity 转换（A733 NPU v3，optimize 目标 **a733**）：

```bash
cd tmp/awnpu_model_zoo-v1.0.0-20260423-f562dd16/examples/mobilenetv2/convert_model/
./convert_model_env.sh
./pegasus_import.sh mobilenetv2-12
./pegasus_quantize.sh mobilenetv2-12 pcq 10
./pegasus_export_ovx_nbg.sh mobilenetv2-12 pcq a733
```

`config_yml.py` 需保持 `ADD_PREPROC_NODE=True`、`PREPROC_TYPE=IMAGE_RGB`、`ADD_POSTPROC_NODE=True`（NPU 内 mean/scale + FP32 输出）。

将导出产物 `mobilenetv2-12_pcq_a733.nb` 复制为 fixture：

```bash
cp mobilenetv2-12_pcq_a733.nb tests/fixtures/integration/awnn/mobilenetv2_classify/mobilenetv2.nb
```

## 构建与安装

```bash
cmake --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release
cmake --build --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --target mobilenetv2_classify_awnn bench_mobilenetv2_classify
bash scripts/install.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release
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
  share/percept/fixtures/integration/awnn/mobilenetv2_classify/mobilenetv2.nb \
  share/percept/fixtures/integration/awnn/mobilenetv2_classify/cat.jpg 5
```

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
  --board orangepi@<ip>:~/deploy_percept
```

对应 `share/percept/tests/test_mobilenetv2_classify_awnn`。Top-5 class_id 真值：`{281, 282, 285, 287, 478}`。

## 性能 benchmark

```bash
# 板端：测试完成后加 --bench
bash scripts/test.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --board orangepi@<ip>:~/deploy_percept --bench 50

# 或 install prefix 本地直接跑
export PERCEPT_ROOT=$PWD/share/percept
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./share/percept/benchmarks/bench_mobilenetv2_classify 50
```
