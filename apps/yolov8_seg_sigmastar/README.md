# yolov8_seg_sigmastar

Sigmastar IPU YOLOv8 实例分割 demo（CoMake Pi D1 / SSD2355 / pcupid）。

使用 `SigmastarEngine` + `YoloV8SegPostProcessSigmastar`（2 路 fused det+coeffs + proto，对齐 OpenDLA `segmentation/yolov8`）+ `drawSegmentationResults`。

板端开放模型：`yolov8n_seg_640x640_ssd2355.img`；默认样图 `bus.jpg`（均在 `share/percept/fixtures/integration/sigmastar/`）。

## 准备 Sigmastar SDK

Find 模块期望：

```text
third_party/armv7l-linux-gnueabihf_comake_pi_d1_ssd2355/sigmastar/release/
```

```bash
mkdir -p third_party/armv7l-linux-gnueabihf_comake_pi_d1_ssd2355/sigmastar
ln -sfn /path/to/Linux_SDK_Release_*/SourceCode/project/release \
  third_party/armv7l-linux-gnueabihf_comake_pi_d1_ssd2355/sigmastar/release
```

并确保 `release/include/mi_ipu.h` 可用。

## 模型与输入

| 文件 | 说明 |
|------|------|
| `yolov8n_seg_640x640_ssd2355.img` | pcupid 开放分割模型（640×640） |
| `bus.jpg` | OpenDLA `source/resource/bus.jpg` |

```bash
cp tmp/Pcupid_DLC173V2.3.3_0918/Linux_SDK_Release_20250918/SourceCode/project/board/pcupid/dla_file/ipu_open_models/segmentation/yolov8n_seg_640x640.img \
  tests/fixtures/integration/sigmastar/yolov8n_seg_640x640_ssd2355.img
cp tmp/Pcupid_DLC173V2.3.3_0918/Linux_SDK_Release_20250918/SourceCode/sdk/verify/opendla/source/resource/bus.jpg \
  tests/fixtures/integration/sigmastar/bus.jpg
```

## 构建

```bash
cmake --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release
cmake --build --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release \
  --target yolov8_seg_sigmastar
```

## 运行（板端）

需要固件默认 `/config/dla/ipu_firmware.bin`。

```bash
cd /customer/deploy_percept
./bin/yolov8_seg_sigmastar \
  ./share/percept/fixtures/integration/sigmastar/yolov8n_seg_640x640_ssd2355.img \
  ./share/percept/fixtures/integration/sigmastar/bus.jpg \
  ./yolov8_seg_sigmastar_out.jpg
```

| 位置参数 | 说明 |
|----------|------|
| `argv[1]` | IPU 模型 `.img`（默认 `./yolov8n_seg_640x640_ssd2355.img`） |
| `argv[2]` | 输入图片（默认 `./bus.jpg`） |
| `argv[3]` | 输出可视化（默认 `./yolov8_seg_sigmastar_out.jpg`） |
