# yolov8_detect_dump_sigmastar

Sigmastar **板端检测 dump**：对目录内图片批量推理，写出 **conf 过滤 + NMS** 后的检测框 JSON。  
**不做精度计算**；PC 侧用同 schema 的 ONNX dump 与比对脚本：

- `scripts/eval_yolov8_detect_dump/eval_onnx_dump.py`
- `scripts/eval_yolov8_detect_dump/compare_dumps.py`

约定：

- dump 时 `--conf` 宜偏低（默认 `0.01`），PC 再按更高 score 上滤以选工作点 / 调 PR
- 预处理与 demo 一致：**plain resize**（非 LetterBox）
- 框坐标：**原图像素** `xyxy`；`path_img` 相对 `--image-dir`

## 构建

```bash
cmake --preset=armv7l-linux-gnueabihf_ssc308qe-release
cmake --build --preset=armv7l-linux-gnueabihf_ssc308qe-release \
  --target yolov8_detect_dump_sigmastar
```

依赖：`nlohmann_json`（`third_party/.../nlohmann_json`）。其它 Sigmastar preset 同理。

产物：`build/.../apps/yolov8_detect_dump_sigmastar/yolov8_detect_dump_sigmastar`

## 运行（板端只 dump）

```bash
cd /customer/deploy_percept
./bin/yolov8_detect_dump_sigmastar \
  --model ./yolov8n.img \
  --image-dir ./val_images \
  --out ./dump_board.json \
  --conf 0.01 \
  --nms 0.5
```

| 参数 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `--model` | 是 | — | Sigmastar `.img` |
| `--image-dir` | 是 | — | 递归扫描 jpg/jpeg/png/bmp |
| `--out` | 否 | `dump_board.json` | 输出 JSON |
| `--conf` | 否 | `0.01` | dump conf |
| `--nms` | 否 | `0.5` | NMS IoU |
| `-h` / `--help` | 否 | — | 用法 |

`max_det` 固定 300，写入 `meta.max_det`。

## JSON schema（摘要）

```json
{
  "meta": {
    "format_version": 1,
    "source": "board",
    "model": "yolov8n.img",
    "dump_conf": 0.01,
    "nms": 0.5,
    "max_det": 300,
    "coord_space": "original_xyxy",
    "preprocess": "resize",
    "model_in_w": 640,
    "model_in_h": 640
  },
  "images": [
    {
      "path_img": "subdir/a.jpg",
      "img_w": 1920,
      "img_h": 1080,
      "detections": [
        {"class_id": 0, "score": 0.9, "x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0}
      ]
    }
  ]
}
```
