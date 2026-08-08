# mobilenetv2_classify_sigmastar

Sigmastar IPU MobileNetV2 分类 demo（CoMake Pi D1 / SSD2355 / pcupid）。

使用 `SigmastarEngine` + `ClassificationPostProcessSigmastar`（INT16 × `fScalar` → Softmax → Top-1），对齐 OpenDLA `classification` 样例。

板端开放模型：`mobilenet_v2_224_ssd2355.img`（`share/percept/fixtures/integration/sigmastar/`）。输出为 ImageNet-1000；OpenDLA demo 源文件名仍为 `mobilenet_v2_224.img`。

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

## 模型与输入

| 文件 | 说明 |
|------|------|
| `mobilenet_v2_224_ssd2355.img` | pcupid 开放分类模型（224×224，ImageNet-1000） |
| `dog.jpg` | 默认输入（与检测 fixture 共用） |

源码树：`tests/fixtures/integration/sigmastar/`；install 后为 `share/percept/fixtures/integration/sigmastar/`。

模型可从 SDK 复制：

```bash
cp tmp/Pcupid_DLC173V2.3.3_0918/Linux_SDK_Release_20250918/SourceCode/project/board/pcupid/dla_file/ipu_open_models/classification/mobilenet_v2_224.img \
  tests/fixtures/integration/sigmastar/mobilenet_v2_224_ssd2355.img
```

## 构建

```bash
cmake --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release
cmake --build --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release \
  --target mobilenetv2_classify_sigmastar
```

产物：`build/.../apps/mobilenetv2_classify_sigmastar/mobilenetv2_classify_sigmastar`

## 运行（板端）

需要：

- install 树：`bin/mobilenetv2_classify_sigmastar` + `lib/libcnpy.so`（及 zlib）
- 模型 `mobilenet_v2_224_ssd2355.img`
- 固件默认 `/config/dla/ipu_firmware.bin`
- 输入图片

```bash
cd /customer/deploy_percept
# 若二进制未带 RPATH，可临时：
# export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./bin/mobilenetv2_classify_sigmastar \
  ./share/percept/fixtures/integration/sigmastar/mobilenet_v2_224_ssd2355.img \
  ./share/percept/fixtures/integration/sigmastar/dog.jpg
```

| 位置参数 | 说明 |
|----------|------|
| `argv[1]` | IPU 模型 `.img`（默认 `./mobilenet_v2_224_ssd2355.img`） |
| `argv[2]` | 输入图片（默认 `./dog.jpg`） |
