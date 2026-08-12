# yolov8_detect_sigmastar

Sigmastar IPU YOLOv8 检测 demo（CoMake Pi D1 / SSD2355 / pcupid）。

使用 `SigmastarEngine` + `YoloV8DetectPostProcessSigmastar`（1/3/5 路输出 decode + NMS）+ `drawDetectionResults`。

板端已验证模型：`yolov8n_640x640_ssd2355.img`（`share/percept/fixtures/integration/sigmastar/`）。

## 准备 Sigmastar SDK

Find 模块期望（只放 `project/release`，对齐 AWNN 的 `awnn/npuruntime`）：

```text
third_party/armv7l-linux-gnueabihf_comake_pi_d1_ssd2355/sigmastar/release/
```

```bash
mkdir -p third_party/armv7l-linux-gnueabihf_comake_pi_d1_ssd2355/sigmastar
ln -sfn /path/to/Linux_SDK_Release_*/SourceCode/project/release \
  third_party/armv7l-linux-gnueabihf_comake_pi_d1_ssd2355/sigmastar/release
```

并确保 `release/include/mi_ipu.h` 可用（若仅有 `include/ipu/pcupid/mi_ipu.h`，需扁平化或软链）。

## 构建

```bash
cmake --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release
cmake --build --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release \
  --target yolov8_detect_sigmastar
```

产物：`build/.../apps/yolov8_detect_sigmastar/yolov8_detect_sigmastar`

## 运行（板端）

需要：

- install 树：`bin/yolov8_detect_sigmastar` + `lib/libcnpy.so`（及 zlib）
- 模型 `yolov8n_640x640_ssd2355.img`
- 固件默认 `/config/dla/ipu_firmware.bin`
- 输入图片

```bash
cd /customer/deploy_percept
# 若二进制未带 RPATH，可临时：
# export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./bin/yolov8_detect_sigmastar \
  ./share/percept/fixtures/integration/sigmastar/yolov8n_640x640_ssd2355.img \
  ./share/percept/fixtures/integration/sigmastar/dog.jpg \
  ./out.jpg
```

| 位置参数 | 说明 |
|----------|------|
| `argv[1]` | IPU 模型 `.img`（默认 `./yolov8n_640x640_ssd2355.img`） |
| `argv[2]` | 输入图片（默认 `./dog.jpg`） |
| `argv[3]` | 输出图片（默认 `./yolov8_detect_sigmastar_out.jpg`） |

可选前缀：`--duration-sec T`（见下）。

## 资源画像（`--duration-sec`）

按墙钟循环推理（预处理只做一次），供 `profiling/percept_monitor` 旁路采样 CPU / RSS / IPU util。`T` 是 **app 循环墙钟**（建议 ≥60）；monitor 跟进程退出。

```bash
cd /customer/tmp/deploy_percept

./bin/yolov8_detect_sigmastar --duration-sec 60 \
  ./share/percept/fixtures/integration/sigmastar/yolov8n_640x640_ssd2355.img \
  ./share/percept/fixtures/integration/sigmastar/dog.jpg &

./profiling/percept_monitor --name yolov8_detect_sigmastar --out /tmp/yolov8_ipu.csv
```

结束时打印 `done: frames=… wall=…s`，并用末帧结果写输出图。
