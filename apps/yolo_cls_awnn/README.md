# yolo_cls_awnn

同进程双模型 demo：YOLOv8 检测 + MobileNetV2 分类（Orange Pi 4 Pro A733）。

一个 `VipLiteRuntime`（`vip_init`），两个 `AwnnEngine`（各加载一个 `.nb`）。主线程每帧串行 `yolo → cls`，NPU 不同时 `vip_run_network`。

两路独立推理（检测用 `dog.jpg`，分类用 `cat.jpg`），**不是**检测框裁块再分类。

## 资源（install → share/percept/fixtures/integration/awnn/）

| 文件 | 说明 |
|------|------|
| `yolov8_a733.nb` | 检测模型 |
| `mobilenetv2_a733.nb` | 分类模型 |
| `dog.jpg` | 检测输入 |
| `cat.jpg` | 分类输入 |

模型准备见 [yolov8_detect_awnn](../yolov8_detect_awnn/README.md)、[mobilenetv2_classify_awnn](../mobilenetv2_classify_awnn/README.md)。

## 构建

```bash
cmake --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release
cmake --build --preset=aarch64-linux-gnu_orange_pi_4_pro_a733-release \
  --target yolo_cls_awnn
bash scripts/build.sh --preset aarch64-linux-gnu_orange_pi_4_pro_a733-release --install
```

## 运行（板端）

```bash
cd ~/deploy_percept
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./bin/yolo_cls_awnn \
  ./share/percept/fixtures/integration/awnn/yolov8_a733.nb \
  ./share/percept/fixtures/integration/awnn/mobilenetv2_a733.nb \
  ./share/percept/fixtures/integration/awnn/dog.jpg \
  ./share/percept/fixtures/integration/awnn/cat.jpg \
  ./yolo_cls_out.jpg
```

| 位置参数 | 默认 |
|----------|------|
| yolo `.nb` | `./yolov8_a733.nb` |
| cls `.nb` | `./mobilenetv2_a733.nb` |
| 检测图 | `./dog.jpg` |
| 分类图 | `./cat.jpg` |
| 检测可视化输出 | `./yolo_cls_awnn_out.jpg` |

可选前缀：`--duration-sec T`。

结束打印 `detection num`、分类 `top1`、以及输出路径。`dog.jpg` 上 detection num 应与 `yolov8_detect_awnn` 一致；`cat.jpg` 上 `top1 class_id` 应与 `mobilenetv2_classify_awnn` 一致（281）。

## 资源画像（`--duration-sec`）

按墙钟循环（预处理只做一次；每帧两路各 run 一次），供 `profiling/percept_monitor` 旁路采样。

**推荐（无参编排）：**

```bash
cd ~/deploy_percept
sh scripts/profile_run.sh
```

`./bin` 有 `*_awnn` 时脚本第三组为本应用。日志在 `var/profile/<时间戳>/`。

手跑等价：

```bash
./bin/yolo_cls_awnn --duration-sec 90 \
  share/percept/fixtures/integration/awnn/yolov8_a733.nb \
  share/percept/fixtures/integration/awnn/mobilenetv2_a733.nb \
  share/percept/fixtures/integration/awnn/dog.jpg \
  share/percept/fixtures/integration/awnn/cat.jpg &

./profiling/percept_monitor --name yolo_cls_awnn --warmup-sec 10
```

对照单模型：RSS 应高于 YOLO 单路（两套权重常驻），且低于两进程 RSS 简单相加。`frames` 为一帧内两路各 Invoke 一次。
