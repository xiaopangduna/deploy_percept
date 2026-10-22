# image_classifiter_from_detect_box_sigmastar

Sigmastar **板端 dump**：按检测框裁剪后做分类，批量写出 float logits CSV。  
**不做精度计算**；ONNX 对齐与混淆矩阵在训练仓完成：

- `deep_learning/src/train/scripts/eval_box_crop_classifier_onnx.py`
- `deep_learning/src/train/scripts/confusion_from_classifier_dump.py`

流程：读 CSV → 原图按 `cx,cy,w,h` × `box_scale` 裁切 → LetterBox(fill=114) → 按模型输入格式打包（BGRA/NV12）→ `INT16 × fScalar`（Softmax 前）→ 写 dump CSV。

标签映射 A：CSV `class_id` `0/1 → gt_id=1`，`2 → gt_id=2`（不合成 background）。

## 构建

```bash
cmake --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release
cmake --build --preset=armv7l-linux-gnueabihf_comake_pi_d1_ssd2355-release \
  --target image_classifiter_from_detect_box_sigmastar
```

SSC308QE 等其它 Sigmastar preset 同理替换 preset 名。

产物：`build/.../apps/image_classifiter_from_detect_box_sigmastar/image_classifiter_from_detect_box_sigmastar`

## 板端数据布局

CSV 内 `path_img` 相对 **CSV 所在目录**，例如：

```text
v301_eye_copy/
  test_v301.csv
  images/test_v301/*.jpg
eye_v007.img
```

## 运行（板端只 dump）

```bash
cd /customer/deploy_percept
./bin/image_classifiter_from_detect_box_sigmastar \
  --model ./eye_v007.img \
  --csv ./v301_eye_copy/test_v301.csv \
  --out ./dump_board.csv \
  --box-scale 1.2
```

| 参数 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `--model` | 是 | — | Sigmastar `.img` |
| `--csv` | 是 | — | 标注 CSV |
| `--out` | 否 | `dump_board.csv` | dump 输出路径 |
| `--box-scale` | 否 | `1.2` | 裁框外扩系数 |
| `-h` / `--help` | 否 | — | 用法 |

Dump 列：`path_img,class_id,gt_id,logit_0,logit_1,logit_2,pred_id`

## PC / 训练侧评估

把 `dump_board.csv` 拷回 PC 后，在 `train` 仓库：

```bash
cd /path/to/deep_learning/src/train

python scripts/eval_box_crop_classifier_onnx.py \
  --model path/to/eye_v007.onnx \
  --csv path/to/v301_eye_copy/test_v301.csv \
  --out dump_onnx.csv

python scripts/confusion_from_classifier_dump.py --dump dump_onnx.csv
python scripts/confusion_from_classifier_dump.py --dump dump_board.csv
```
