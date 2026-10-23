# eval_yolov8_detect_dump

PC 侧：与板端 [`yolov8_detect_dump_sigmastar`](../../apps/yolov8_detect_dump_sigmastar/README.md) 同 schema 的 ONNX dump，以及板端↔ONNX 一致性比对。

预处理与板端一致：**plain resize**（非 LetterBox），框为原图像素 `xyxy`。

## 依赖

```bash
pip install onnxruntime opencv-python numpy
```

## ONNX dump

```bash
python scripts/eval_yolov8_detect_dump/eval_onnx_dump.py \
  --model path/to/yolov8n.onnx \
  --image-dir path/to/val_images \
  --out dump_onnx.json \
  --conf 0.01 \
  --nms 0.5
```

`--conf` / `--nms` 应与板端 dump 一致。ONNX 需为常见 Ultralytics 导出（输出 `(1, 4+nc, N)` 或 `(1, N, 4+nc)`）。

## 比对

```bash
python scripts/eval_yolov8_detect_dump/compare_dumps.py \
  --board dump_board.json \
  --onnx dump_onnx.json \
  --score-thresh 0.5 \
  --iou-thresh 0.5
```

`--score-thresh` 在 dump 之上上滤（应 ≥ dump 时的 `dump_conf`）。按同 `class_id` + IoU 贪心匹配，输出 match 率、mean IoU、mean |Δscore|。
