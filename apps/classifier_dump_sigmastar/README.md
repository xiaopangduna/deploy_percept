# classifier_dump_sigmastar

Sigmastar **板端 dump**：按 CSV 做分类，写出一个 JSON。`meta` 里有这次运行的参数和 `acc1` / `acc5`，`samples` 里是每条成功样本的 logit。

混淆矩阵在训练仓计算：

- `deep_learning/src/train/scripts/confusion_from_classifier_dump.py`

流程：读 CSV → 按模型决定读图通道 → 整图，或在 `--box-scale > 0` 且 CSV 有框时按 `cx,cy,w,h` × `box_scale` 裁切 → 按 `--prep` 缩放到模型尺寸 → 按 `.img` 的 `eElmFormat` 打包 → `元素 × fScalar`（Softmax 前）→ 写 JSON。

输入格式自适应（由模型描述决定，无需改 CLI）。读图跟模型走：灰度在 LetterBox 之前就是单通道；彩色仍按 BGR 读入。

| 量化 `input_formats` | 常见 `eElmFormat` | 读图与打包 |
|----------------------|-------------------|------------|
| GRAY | `GRAY` | `IMREAD_GRAYSCALE`，单通道紧密排列 |
| GRAY（旧 SDK 落成 U8） | `U8` 且 `H×W <= buf < H×W×3` | 同上 |
| BGRA | `ARGB8888` | `IMREAD_COLOR`，BGR→BGRA，stride 16 对齐 |
| RGBA | `ABGR8888` | `IMREAD_COLOR`，BGR→RGBA，stride 16 对齐 |
| RGB | `RGB888` 或旧 SDK `U8`(buf≈H×W×3) | `IMREAD_COLOR`，BGR→RGB，**紧密排列** |
| BGR | `BGR888` / `U8` | `IMREAD_COLOR`，BGR 紧密排列 |
| YUV_NV12 | `NV12` | `IMREAD_COLOR`，BGR→NV12 |

启动时会打印 `input elm : …` 和 `pack mode : …`，便于确认新量化模型实际走了哪条路径。

整图还是裁框：

| 条件 | 行为 |
|------|------|
| 未传 `--box-scale`，或值 `<= 0` | 整图，再按 `--prep` 缩放 |
| `--box-scale > 0`，且 CSV 有 `cx,cy,w,h` | 先按归一化框裁切，再按 `--prep` 缩放 |
| `--box-scale > 0`，但框列缺少 | 整图，并在 stderr 说明 |

`--prep`：

| 值 | 几何 |
|----|------|
| `letterbox`（默认） | 保持宽高比，不足处填 114 |
| `resize` | 直接 `INTER_LINEAR` 拉到模型 `H×W` |
| `imagenet` | 短边缩到 `floor(min(H,W) / 0.875)`，再中心裁到模型尺寸。224 输入就是短边 256、中心裁 224 |

像素保持原值。要不要做 `(像素 - mean) / std` 在转 `.img` 时由 `input_config` 决定；ONNX 侧在 `train/exports/sigmastar/mobilenet_v2/verify/multi_image_accuracy_dump_onnx_json.py` 里做。这个程序不减均值、不除标准差。

`cx,cy,w,h` 是相对原图宽高的归一化值。裁框时写出的样本里仍是这个值，还没乘 `box_scale`。

`acc1`、`acc5` 只比较 CSV 的 `class_id` 和 `pred_id`。标签映射不在这里做，放到别的脚本里。

## 构建

需要 `nlohmann_json`。

```bash
cmake --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release
cmake --build --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release \
  --target classifier_dump_sigmastar
```

SSC308QE 等其它 Sigmastar preset 同理替换 preset 名。

产物：`build/.../apps/classifier_dump_sigmastar/classifier_dump_sigmastar`

## 板端数据布局

CSV 内 `path_img` 相对 **CSV 所在目录**，例如：

```text
v301_eye_copy/
  test_v301.csv
  images/test_v301/*.jpg
eye_v007.img
```

## 运行

```bash
cd /customer/deploy_percept
./bin/classifier_dump_sigmastar \
  --model ./eye_v007.img \
  --csv ./v301_eye_copy/test_v301.csv \
  --box-scale 1.2
```

不传 `--out` 时，输出是模型文件名去掉扩展名后加 `.json`。上面这条因为带了 `--box-scale 1.2`，并且 CSV 有框，会裁框后写到当前目录的 `eye_v007.json`。不传 `--box-scale` 则整图。

| 参数 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `--model` | 是 | — | Sigmastar `.img` |
| `--csv` | 是 | — | 标注 CSV |
| `--out` | 否 | `<模型文件名>.json` | dump 输出路径 |
| `--box-scale` | 否 | 整图 | `> 0` 且 CSV 有 `cx,cy,w,h` 时作为外扩系数裁框；`<= 0` 或不传则整图 |
| `--prep` | 否 | `letterbox` | `letterbox`、`resize`、`imagenet` |
| `-h` / `--help` | 否 | — | 用法 |

`--prep imagenet` 只做验证集的几何：短边 256、中心裁 224。像素除以 255 和 mean/std 不在这里。

类别数是分类头输出张量的元素个数。`acc1` 是 `pred_id == class_id` 的比例，`acc5` 是 `class_id` 落在 logit 最大的 `top_k` 个里的比例，`top_k = min(5, num_classes)`。分母是成功样本数。`class_id` 不在 `0 .. num_classes-1` 时这条算错，仍然写入 `samples`。这两个数写入 `meta`，并打到 stdout。没有成功样本时 `acc1` / `acc5` 为 `null`，进程返回 1。

读图失败、推理失败的样本不进 `samples`，只计入 `meta.skip`。原因打在 stderr。

## JSON

```json
{
  "meta": {
    "format_version": 1,
    "source": "board",
    "model": "eye_v007.img",
    "csv": "v301_eye_copy/test_v301.csv",
    "mode": "box",
    "box_scale": 1.2,
    "num_classes": 3,
    "output_shape": [1, 3],
    "output_fscalar": 0.000183,
    "model_h": 224,
    "model_w": 224,
    "prep": "letterbox",
    "letterbox_fill": 114,
    "elm_format": 6,
    "elm_format_name": "GRAY",
    "pack_mode": "gray_tight",
    "ok": 1000,
    "skip": 3,
    "top_k": 3,
    "acc1": 0.912,
    "acc5": 1.0
  },
  "samples": [
    {
      "row": 0,
      "path_img": "images/a.jpg",
      "class_id": 0,
      "cx": 0.51,
      "cy": 0.48,
      "w": 0.12,
      "h": 0.09,
      "logits": [1.2, -0.4, 0.3],
      "pred_id": 0
    }
  ]
}
```

整图模式的 `meta` 没有 `box_scale`，`mode` 为 `"image"`。样本没有 `cx,cy,w,h`。

`row` 是该样本在读入的 CSV 数据行里的下标，从 0 起。空行和字段不够的行不计入。`logits` 是 softmax 之前的值，长度等于 `num_classes`。`pred_id` 是 argmax。

推理时样本先写入 `<输出名>.samples.tmp`，结束后拼成最终 JSON 再删掉临时文件。中途中断时临时文件里是已经完成的样本。

## PC

把 `eye_v007.json` 拷回 PC 后，在 `train` 仓库：

```bash
cd /path/to/deep_learning/src/train

python scripts/confusion_from_classifier_dump.py --dump eye_v007.json
```

`meta.acc1` 是板上用 `class_id` 和 `pred_id` 算的 top-1。混淆矩阵、以及另外一套标签映射，用 `samples` 里的 `class_id,pred_id` 在别的脚本里算。ONNX 侧 `eval_box_crop_classifier_onnx.py` 仍写带 `gt_id` 的 CSV，`confusion_from_classifier_dump.py` 读 CSV 时用 `gt_id`，读这份 JSON 时用 `class_id`。
