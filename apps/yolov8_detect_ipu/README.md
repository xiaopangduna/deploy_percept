# yolov8_detect_ipu

Sigmastar IPU YOLOv8 检测 demo（CoMake Pi D1 / SSD2355 / pcupid）。

源码从 OpenDLA sample 复现：

`tmp/.../sdk/verify/opendla/source/detection/yolov8` + `source/utils`。

当前为 **原始 MI_IPU API 直调**（尚未接入 `SigmastarIpuEngine` / 框架后处理）。板端已验证模型：`yolov8n_640x640.img`。

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
  --target yolov8_detect_ipu
```

或：

```bash
bash scripts/build.sh --preset armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release
```

产物：`build/.../apps/yolov8_detect_ipu/yolov8_detect_ipu`

## 运行（板端）

需要：

- 可执行文件 `yolov8_detect_ipu`
- 模型 `yolov8n_640x640.img`（SDK：`project/board/pcupid/dla_file/ipu_open_models/detection/`）
- 固件默认 `/config/dla/ipu_firmware.bin`
- 输入图片

```bash
./yolov8_detect_ipu -i "./dog.png" -m ./yolov8n_640x640.img -t 0.5
```

| 参数 | 说明 |
|------|------|
| `-i` | 输入图片（支持 glob） |
| `-m` | IPU 模型 `.img` |
| `-t` | 置信度阈值（默认 0.5） |

结果写到 `./output/<timestamp>/`。
