# yolov8_detect_awnn

全志 VIPLite YOLOv8 检测（Orange Pi 4 Pro A733）。推理经 `deploy_percept::engine::AwnnEngine`，后处理见 `YoloV8DetectPostProcessAwnn`（与 `tmp/awnpu_model_zoo-.../examples/yolov8` 六路 FP32 输出一致）。

## 资源（install → share/percept/apps/yolov8_detect_awnn/）

| 文件 | 说明 |
|------|------|
| `yolov8.nb` | A733 模型（`yolov8n_6_uint8_a733.nb` 的 symlink） |
| `dog.jpg` | 默认输入图 |

## 模型训练与转换

完整链路：

```
PC 训练/下载 (Ultralytics) → 导出 ONNX → onnxsim 固定尺寸 → onnx_extract 裁剪 6 路 head → Acuity 量化/编译 → yolov8.nb → 板端 AWNN 推理
```

本 demo 的模型 I/O 约定：

- **输入**：UINT8 RGB HWC 640×640（resize；转换时 `ADD_PREPROC_NODE=True`, `PREPROC_TYPE=IMAGE_RGB`）
- **输出**：六路 **FP32**（stride 8/16/32 各 grid + score），后处理在 CPU 完成；检测框坐标在 **模型输入尺寸**（640×640）空间

参考 `tmp/awnpu_model_zoo-v1.0.0-20260423-f562dd16/examples/yolov8/`：

1. 导出 ONNX：`ultralytics` v8.1.0，`model.export(format='onnx', dynamic=True, opset=11)`
2. 固定 shape：`python3 -m onnxsim yolov8n.onnx yolov8n_640_sim.onnx --input-shape=1,3,640,640`
3. 裁剪 6 路 head：`convert_model/python/onnx_extract.py` → `yolov8n_6.onnx`
4. Acuity 转换（A733 对应 **mr536/t536** optimize 目标）：

```bash
cd tmp/awnpu_model_zoo-v1.0.0-20260423-f562dd16/examples/yolov8/convert_model/
./convert_model_env.sh
./pegasus_import.sh yolov8n_6
./pegasus_quantize.sh yolov8n_6 uint8 12
./pegasus_export_ovx_nbg.sh yolov8n_6 uint8 mr536
```

`config_yml.py` 中需保持 `ADD_PREPROC_NODE=True`、`ADD_POSTPROC_NODE=True`（量化输出转 FP32）。

### 替换 install 包内模型

```bash
cp yolov8n_6_uint8_mr536.nb /path/to/install/share/percept/apps/yolov8_detect_awnn/yolov8.nb
```

## 构建与安装

```bash
cmake --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release -DBUILD_AWNN_APPS=ON
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

默认输出：`share/percept/apps/yolov8_detect_awnn/yolov8_detect_awnn_out.jpg`

也可显式指定路径：

```bash
./bin/yolov8_detect_awnn share/percept/apps/yolov8_detect_awnn/yolov8.nb \
  share/percept/apps/yolov8_detect_awnn/dog.jpg /tmp/out.jpg
```

## 模型 I/O

- 输入：UINT8 RGB HWC 640×640（resize）
- 输出：6 路 FP32（stride 8/16/32 × grid/score），后处理用 `YoloV8DetectPostProcessAwnn`；框坐标为模型输入像素空间
