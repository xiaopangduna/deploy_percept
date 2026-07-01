# yolov8_detect_awnn

全志 VIPLite YOLOv8 检测（Orange Pi 4 Pro A733）。推理经 `deploy_percept::engine::AwnnEngine`，后处理见 `YoloV8DetectPostProcessAwnn`（与 `tmp/awnpu_model_zoo-.../examples/yolov8` 六路 FP32 输出一致）。

## 资源（install → share/percept/apps/yolov8_detect_awnn/）

| 文件 | 说明 |
|------|------|
| `yolov8.nb` | A733 模型（`yolov8n_6_uint8_a733.nb` 的 symlink） |
| `dog.jpg` | 默认输入图（768×576） |

## 模型训练与转换

完整链路：

```
PC 训练/下载 (Ultralytics) → 导出 ONNX → onnxsim 固定尺寸 → onnx_extract 裁剪 6 路 head → Acuity 量化/编译 → yolov8.nb → 板端 AWNN 推理
```

参考 `tmp/awnpu_model_zoo-v1.0.0-20260423-f562dd16/examples/yolov8/`：

1. 导出 ONNX：`ultralytics` v8.1.0，`model.export(format='onnx', dynamic=True, opset=11)`
2. 固定 shape：`python3 -m onnxsim yolov8n.onnx yolov8n_640_sim.onnx --input-shape=1,3,640,640`
3. 裁剪 6 路 head：`convert_model/python/onnx_extract.py` → `yolov8n_6.onnx`
4. Acuity 转换（A733 NPU v3，optimize 目标 **`a733`** 或 **`mr536`/`t536`**）：

```bash
cd tmp/awnpu_model_zoo-v1.0.0-20260423-f562dd16/examples/yolov8/convert_model/
./convert_model_env.sh
./pegasus_import.sh yolov8n_6
./pegasus_quantize.sh yolov8n_6 uint8 12
./pegasus_export_ovx_nbg.sh yolov8n_6 uint8 a733
```

`config_yml.py` 中需保持 `ADD_PREPROC_NODE=True`、`ADD_POSTPROC_NODE=True`（量化输出转 FP32）。转换时 `PREPROC_TYPE=IMAGE_RGB`，与 demo 的 RGB HWC 输入一致。

### 替换 install 包内模型

```bash
cp yolov8n_6_uint8_a733.nb /path/to/install/share/percept/apps/yolov8_detect_awnn/yolov8.nb
# 板端 rsync install 树后重跑 demo / test.sh
```

## 构建与安装

```bash
cmake --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release
cmake --build --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release --target yolov8_detect_awnn
bash scripts/install.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release
```

## 运行

**install 树（板端）：**

```bash
cd ~/deploy_percept
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./bin/yolov8_detect_awnn
```

未设置 `PERCEPT_ROOT` 时，程序从当前目录或可执行文件路径查找 `share/percept/apps/`。也可显式设置：

```bash
export PERCEPT_ROOT=$PWD/share/percept
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
```

默认输出：`share/percept/apps/yolov8_detect_awnn/yolov8_detect_awnn_out.jpg`

显式指定路径：

```bash
./bin/yolov8_detect_awnn share/percept/apps/yolov8_detect_awnn/yolov8.nb \
  share/percept/apps/yolov8_detect_awnn/dog.jpg /tmp/out.jpg
```

**正确性回归（板端）：**

```bash
bash scripts/test.sh --board orangepi@<ip>:~/deploy_percept
```

对应 `share/percept/tests/test_yolov8_detect_awnn`（integration test）。

## 预处理

demo 与 integration test 使用相同流程：

1. 读取 BGR 原图
2. **直接 resize** 到模型输入尺寸 640×640（无 letterbox）
3. BGR → RGB，按 **HWC 逐像素交错** 写入 VIP input buffer

若需将检测框映射回原图，请在应用层自行做 resize 逆变换；后处理 **不** 做原图投影。

## 模型 I/O

### 输入

| 项 | 值 |
|----|-----|
| dtype | UINT8 |
| VIP `input_sizes` | `[C, H, W, N] = [3, 640, 640, 1]` |
| buffer 布局 | RGB **HWC** 交错（`sizes[1]`=H，`sizes[2]`=W） |
| 字节数 | 1 228 800（640×640×3） |

与 YOLOv5 demo 不同：yolov8 为 **HWC**，yolov5 为 **NCHW**（`[W, H, C, N]`）。

### 输出（6 路 FP32）

| stride | grid tensor | score tensor |
|--------|-------------|--------------|
| 8 | `[80, 80, 64, 1]` | `[80, 80, 80, 1]` |
| 16 | `[40, 40, 64, 1]` | `[40, 40, 80, 1]` |
| 32 | `[20, 20, 64, 1]` | `[20, 20, 80, 1]` |

后处理 `YoloV8DetectPostProcessAwnn` 在 CPU 完成 DFL 解码、sigmoid、NMS。检测框坐标落在 **模型输入像素空间**（640×640）。

demo 在 **resize 后的 640×640 图** 上画框，而非原图。

### dog.jpg 板端标定（A733 + yolov8.nb，640×640 模型坐标）

| cls | COCO | box [left, top, right, bottom] | prob |
|-----|------|--------------------------------|------|
| 16 | dog | [110, 248, 258, 601] | 0.8922 |
| 1 | bicycle | [100, 148, 474, 467] | 0.7918 |
| 2 | car | [387, 83, 578, 191] | 0.6005 |

integration test 容差 ±4px，见 `tests/integration/awnn/test_yolov8_detect_awnn.cpp`。
