# dump_img_output_from_image

Sigmastar **板端张量 dump**：一张图进 `.img`，把全部输出反量化成 float32，写进一个 NPZ。

不做 softmax、NMS，也不做转置。轴顺序就是 IPU buffer 的顺序。和 ONNX 的 shape 是否一致，等看到两边文件再改。

PC 侧比对放在训练仓库，按模型分开。MobileNetV2：

- `deep_learning/src/train/exports/sigmastar/mobilenet_v2/dump_onnx_output_from_image.py`
- `deep_learning/src/train/exports/sigmastar/mobilenet_v2/compare_img_and_onnx_dumps.py`

输入格式由 `.img` 的 `eElmFormat` 决定，规则与 `classifier_dump_sigmastar` 相同。图已经是模型 `H×W` 时原样使用；否则 `INTER_LINEAR` 缩放到模型输入。推荐直接喂一张已经缩好的 PNG。

反量化：`float32 = 元素 × fScalar`（zero point 按 0）。`zero_point != 0` 时仍按这个式子写，并在 stderr 打出来。

## 构建

```bash
cmake --preset=armv7l-linux-gnueabihf_ssc308qe-release
cmake --build --preset=armv7l-linux-gnueabihf_ssc308qe-release \
  --target dump_img_output_from_image
```

需要 `nlohmann_json`。其它 Sigmastar preset 同样替换 preset 名。

## 运行

```bash
cd /customer/deploy_percept
./bin/dump_img_output_from_image \
  --model ./model.img \
  --image ./input.png \
  --out ./dump_board.npz
```

| 参数 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `--model` | 是 | — | Sigmastar `.img` |
| `--image` | 是 | — | 一张图，优先已经是模型 `H×W` 的 PNG |
| `--out` | 否 | `dump_board.npz` | 输出 NPZ |
| `-h` / `--help` | 否 | — | 用法 |

## NPZ

| 键 | 内容 |
|----|------|
| `meta` | JSON 的 uint8 字节。含 `input_sha256`、`packed_sha256`、`resized`、`pack_mode`、每个输出的 `shape` / `src_dtype` / `fscalar` / `zero_point` |
| `output_0` … | float32，C-order，shape 与 IPU 描述一致 |

元素个数是 shape 乘积。`u32BufSize` 里多出来的对齐字节不写入。

## PC

在 `train/exports/sigmastar/mobilenet_v2` 下先跑 `dump_onnx_output_from_image.py`，再用 `compare_img_and_onnx_dumps.py` 和板端 `dump_board.npz` 比。灰度传 `--mean 0 --std 255`，RGB 可省略 `--mean` 和 `--std`。细节见该目录 README。
