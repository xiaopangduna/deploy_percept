# yolov5_detect_awnn

全志 VIPLite YOLOv5 检测（Orange Pi 4 Pro A733）。推理经 `deploy_percept::engine::AwnnEngine`。

## 资源（install → share/percept/apps/yolov5_detect_awnn/）

| 文件 | 说明 |
|------|------|
| `yolov5.nb` | A733 模型 |
| `dog.jpg` | 默认输入图 |

install 包内 `yolov5.nb` 为 **COCO 预训练 YOLOv5s** 经 Acuity 转换后的 NBG 模型，可直接跑 demo。若要换自己的数据集，见下文。
https://docs.aw-ol.com/v853/npu/npu_yolov5/
https://docs.100ask.net/vision/docs/V853/part6/06-1_Yolov5ModelDeployment

## 模型训练与转换

`.nb` 不能在板端或 NPU 上直接「训练」得到，完整链路如下：

```
PC 训练 (PyTorch) → 导出 ONNX → onnxsim 固定尺寸 → Acuity 量化/编译 → yolov5.nb → 板端 AWNN 推理
```

本 demo 的模型 I/O 约定：**输入** UINT8 NCHW 640×640；**输出** 三检测头 FP32（stride 8/16/32），后处理见 `YoloV5DetectPostProcessAwnn`。转换时需导出 **三个 raw head**，不要把 NMS/Decode 放进 NPU 图里。

Acuity Toolkit 与 VivanteIDE 需自行准备（全志/Radxa SDK 或 Docker），`tmp/ai-sdk` 只提供转换脚本与示例配置，不含工具链本体。A733 对应 NPU **v3**（`VIP9000NANODI_PLUS`，与 `mr536`/`t536` 相同 optimize 目标）。

### 1. 训练（PC，Ultralytics YOLOv5）

```bash
git clone -b v6.0 https://github.com/ultralytics/yolov5.git   # 与官方 release / ai-sdk 示例一致
cd yolov5
pip install -r requirements.txt

# 准备 data.yaml（train/val 路径、nc、names）
python train.py --data your_data.yaml --weights yolov5s.pt --epochs 100 --img 640
# 权重：runs/train/exp/weights/best.pt
```

也可跳过训练，直接使用 [YOLOv5 v6.0 Release](https://github.com/ultralytics/yolov5/releases/tag/v6.0) 中的 `yolov5s.pt` / `yolov5s.onnx`（COCO 预训练）。

### 2. 导出 ONNX

Release 里的 `yolov5s.onnx` 为 **动态 shape + 单路 `output`（含后处理）**，不适合本 demo。需 **train 模式** 导出三头，再固定输入尺寸：

```bash
cd yolov5
python export.py \
  --weights runs/train/exp/weights/best.pt \
  --include onnx \
  --train \
  --dynamic \
  --imgsz 640 \
  --device cpu
# 预训练权重可将 best.pt 换为 yolov5s.pt；得到 best.onnx

pip install onnxsim onnxruntime
python -m onnxsim best.onnx yolov5s-sim.onnx --input-shape 1,3,640,640
```

用 [Netron](https://netron.app) 打开 `yolov5s-sim.onnx`，确认输入名为 `images`、输出为 **3 个** detection head（无合并后的 `output`）。若自训模型输出节点 ID 与示例不同，下一步 `inputs_outputs.txt` 需按 Netron 修改（官方 v6.0 + ai-sdk 示例为 `350 498 646`）。

### 3. Acuity 转 `.nb`（PC）

参考 `tmp/ai-sdk/models/yolov5s-sim/` 与 `tmp/ai-sdk/scripts/`：

```bash
export ACUITY_PATH=/path/to/acuity-toolkit/bin
export VIV_SDK=/path/to/VivanteIDE/cmdtools

mkdir -p yolov5s-sim && cd yolov5s-sim
cp /path/to/yolov5s-sim.onnx .
cp /path/to/tmp/ai-sdk/models/yolov5s-sim/{inputs_outputs.txt,channel_mean_value.txt,dataset.txt,convert_export.sh} .
# dataset.txt：列出 20~100 张校准图路径；channel_mean_value.txt 默认 0 0 0 0.00392157 (1/255)

source /path/to/tmp/ai-sdk/models/env.sh v3
./convert_export.sh yolov5s-sim uint8 mr536
# 产物：yolov5s-sim_uint8.nb（或 wksp/.../network_binary.nb）
```

分步执行时：`pegasus_import.sh` → 检查/改 `*_inputmeta.yml` → `pegasus_quantize.sh yolov5s-sim uint8` → `pegasus_export_ovx_nbg.sh`（需 `--pack-nbg-unify`）。详见 `tmp/ai-sdk/models/ReadMe.txt`、`tmp/ai-sdk/scripts/README.md`。

外部文档：[全志 V853 YOLOv5 NPU 转换](https://v853.docs.aw-ol.com/npu/npu_yolov5/)、[Radxa A733 YOLOv5](https://docs.radxa.com/cubie/a7s/app-dev/npu-dev/cubie-yolov5)。

### 4. 替换 install 包内模型

```bash
cp yolov5s-sim_uint8.nb /path/to/install/share/percept/apps/yolov5_detect_awnn/yolov5.nb
# 板端 rsync install 树后重跑 demo / test.sh
```

转换后可用 `tmp/ai-sdk/tools/nbinfo` 查看 `.nb` 输入输出是否与上文「模型 I/O」一致。

## 构建与安装

```bash
cmake --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  -DBUILD_AWNN_APPS=ON -DENABLE_BENCHMARKS=ON
cmake --build --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --target yolov5_detect_awnn bench_yolov5_post_process
bash scripts/install.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release
```

## 运行

**install 树（板端）— demo：**

```bash
cd ~/deploy_percept
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./bin/yolov5_detect_awnn
```

未设置 `PERCEPT_ROOT` 时，程序会从当前目录或可执行文件路径自动查找 `share/percept/apps/`。
也可显式设置：`export PERCEPT_ROOT=$PWD/share/percept`

**必须**保证 VIPLite 动态库来自 install 的 `lib/`（与构建版本一致）：

```bash
ls -la lib/libNBGlinker.so lib/libVIPhal.so
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH   # 建议始终设置
```

若出现 `Segmentation fault` 且只打印 VIPLite 版本号，常见原因：

1. `lib/` 未同步或混用了系统里其它版本的 `libVIPhal.so`
2. 可执行文件与 `libdeploy_percept.a` 不同步 — 请完整 rebuild + install + rsync

诊断日志（stderr）会逐步打印 `AwnnEngine: vip_create_network ok` 等；请记下**最后一条**再反馈。

默认输出：`share/percept/apps/yolov5_detect_awnn/yolov5_detect_awnn_out.jpg`

**性能 benchmark（install 后，与测试同一脚本）：**

```bash
# 板端：先跑正确性测试，再加 --bench 跑性能
bash scripts/test.sh --board orangepi@<ip>:~/deploy_percept
bash scripts/test.sh --board orangepi@<ip>:~/deploy_percept --bench 50

# 或 install prefix 本地直接跑 benchmark
export PERCEPT_ROOT=$PWD/share/percept
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./share/percept/benchmarks/bench_yolov5_post_process 50
```

正确性回归见 `share/percept/tests/test_yolov5_detect_awnn`（integration test）。

## 模型 I/O

- 输入：UINT8 NCHW 640×640
- 输出：`yolov5.nb` 为三头 **FP32**（stride 8/16/32），后处理用 `YoloV5DetectPostProcessAwnn`
