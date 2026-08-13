# yolo_cls_sigmastar

同进程双模型 demo：YOLOv8 检测 + MobileNetV2 分类（CoMake Pi D1 / SSD2355 / pcupid）。

一个 `SigmastarRuntime`（`CreateDevice` 的 variable buffer 取两模型 **max**），两个 `SigmastarEngine`（各 `CreateCHN`）。主线程每帧串行 `yolo → cls`，IPU 不同时 Invoke。

## 构建

```bash
cmake --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release
cmake --build --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release \
  --target yolo_cls_sigmastar
```

产物：`build/.../apps/yolo_cls_sigmastar/yolo_cls_sigmastar`

SDK / 固件准备同 `yolov8_detect_sigmastar`。

## 运行（板端）

```bash
cd /customer/tmp/deploy_percept
./bin/yolo_cls_sigmastar \
  ./share/percept/fixtures/integration/sigmastar/yolov8n_640x640_ssd2355.img \
  ./share/percept/fixtures/integration/sigmastar/mobilenet_v2_224_ssd2355.img \
  ./share/percept/fixtures/integration/sigmastar/dog.jpg \
  ./share/percept/fixtures/integration/sigmastar/apple.jpg \
  ./yolo_cls_out.jpg
```

| 位置参数 | 默认 |
|----------|------|
| yolo `.img` | `./yolov8n_640x640_ssd2355.img` |
| cls `.img` | `./mobilenet_v2_224_ssd2355.img` |
| 检测图 | `./dog.jpg` |
| 分类图 | `./apple.jpg` |
| 检测可视化输出 | `./yolo_cls_sigmastar_out.jpg` |

可选前缀：`--duration-sec T`。

启动时打印 `CreateDevice var`（应约为 YOLO 的 3.45MB，不是两路 var 之和）。结束打印 `detection num`、分类 `top1`、以及输出路径。

## 资源画像

**推荐（无参编排）：**

```bash
cd /customer/tmp/deploy_percept
sh scripts/profile_run.sh
```

脚本第三组为 `./bin/yolo_cls_sigmastar --duration-sec 90` + fixture，monitor `--name yolo_cls_sigmastar --warmup-sec 10`。日志在 `var/profile/<时间戳>/`。

手跑等价：

```bash
./bin/yolo_cls_sigmastar --duration-sec 90 \
  share/percept/fixtures/integration/sigmastar/yolov8n_640x640_ssd2355.img \
  share/percept/fixtures/integration/sigmastar/mobilenet_v2_224_ssd2355.img \
  share/percept/fixtures/integration/sigmastar/dog.jpg \
  share/percept/fixtures/integration/sigmastar/apple.jpg &

./profiling/percept_monitor --name yolo_cls_sigmastar --warmup-sec 10
```

对照单模型：`mma_used` 应高于 YOLO 的 ~36.7MB，且低于两路简单相加（Device scratch 只计 max）。`frames` 为一帧内两路各 Invoke 一次。
