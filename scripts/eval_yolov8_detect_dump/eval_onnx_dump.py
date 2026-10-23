#!/usr/bin/env python3
"""YOLOv8 ONNX dump：与板端同 schema 的 JSON（plain resize + conf + NMS）。

依赖：onnxruntime, opencv-python, numpy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FORMAT_VERSION = 1
MAX_DET = 300


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump YOLOv8 ONNX detections to JSON")
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--image-dir", required=True, help="Image directory (recursive)")
    p.add_argument("--out", default="dump_onnx.json", help="Output JSON path")
    p.add_argument("--conf", type=float, default=0.01, help="Dump conf threshold")
    p.add_argument("--nms", type=float, default=0.5, help="NMS IoU threshold")
    return p.parse_args()


def collect_images(image_dir: Path) -> list[str]:
    rels: list[str] = []
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            rels.append(p.relative_to(image_dir).as_posix())
    rels.sort()
    return rels


def letterbox_not_used_resize(bgr: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    """Match board demo: plain cv2.resize (no letterbox)."""
    return cv2.resize(bgr, (model_w, model_h), interpolation=cv2.INTER_LINEAR)


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_thresh: float,
    max_det: int,
) -> list[int]:
    if boxes.size == 0:
        return []
    # Per-class NMS via OpenCV
    keep: list[int] = []
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        if idx.size == 0:
            continue
        b = boxes[idx].tolist()
        s = scores[idx].tolist()
        picked = cv2.dnn.NMSBoxes(b, s, score_threshold=0.0, nms_threshold=iou_thresh)
        if len(picked) == 0:
            continue
        for j in np.array(picked).reshape(-1):
            keep.append(int(idx[int(j)]))
    keep.sort(key=lambda i: float(scores[i]), reverse=True)
    return keep[:max_det]


def decode_yolov8(
    output: np.ndarray,
    conf: float,
    nms: float,
    model_w: int,
    model_h: int,
    img_w: int,
    img_h: int,
) -> list[dict[str, Any]]:
    """Decode common YOLOv8 ONNX output to original-pixel xyxy.

    Accepts shapes:
      - (1, C, N) with C = 4 + num_classes  (Ultralytics export)
      - (1, N, C)
    """
    x = np.asarray(output, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"unexpected ONNX output shape: {output.shape}")

    # Make (N, C)
    if x.shape[0] < x.shape[1] and x.shape[0] <= 512:
        # likely (C, N)
        x = x.T

    if x.shape[1] < 5:
        raise ValueError(f"unexpected feature dim: {x.shape}")

    boxes_xywh = x[:, :4]
    cls_scores = x[:, 4:]
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    mask = scores >= conf
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    if boxes_xywh.size == 0:
        return []

    # xywh center -> xyxy in model space
    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms_xyxy(boxes, scores, class_ids, nms, MAX_DET)
    sx = float(img_w) / float(model_w)
    sy = float(img_h) / float(model_h)

    dets: list[dict[str, Any]] = []
    for i in keep:
        dets.append(
            {
                "class_id": int(class_ids[i]),
                "score": float(scores[i]),
                "x1": float(boxes[i, 0] * sx),
                "y1": float(boxes[i, 1] * sy),
                "x2": float(boxes[i, 2] * sx),
                "y2": float(boxes[i, 3] * sy),
            }
        )
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


def main() -> int:
    args = parse_args()
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is required: pip install onnxruntime", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    image_dir = Path(args.image_dir).resolve()
    if not model_path.is_file():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1
    if not image_dir.is_dir():
        print(f"image-dir not a directory: {image_dir}", file=sys.stderr)
        return 1

    rels = collect_images(image_dir)
    if not rels:
        print(f"no images under: {image_dir}", file=sys.stderr)
        return 1

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    in_name = in_meta.name
    shape = in_meta.shape  # NCHW
    # expect [1, 3, H, W] or dynamic
    model_h = int(shape[2]) if isinstance(shape[2], int) else 640
    model_w = int(shape[3]) if isinstance(shape[3], int) else 640
    out_name = sess.get_outputs()[0].name

    root: dict[str, Any] = {
        "meta": {
            "format_version": FORMAT_VERSION,
            "source": "onnx",
            "model": model_path.name,
            "dump_conf": float(args.conf),
            "nms": float(args.nms),
            "max_det": MAX_DET,
            "coord_space": "original_xyxy",
            "preprocess": "resize",
            "model_in_w": model_w,
            "model_in_h": model_h,
        },
        "images": [],
    }

    print(f"eval_onnx_dump: images={len(rels)} model={model_w}x{model_h}")
    for i, rel in enumerate(rels):
        abs_path = image_dir / rel
        bgr = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[{i+1}/{len(rels)}] failed to read: {rel}", file=sys.stderr)
            continue
        img_h, img_w = bgr.shape[:2]
        resized = letterbox_not_used_resize(bgr, model_w, model_h)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        outs = sess.run([out_name], {in_name: tensor})
        dets = decode_yolov8(
            outs[0], args.conf, args.nms, model_w, model_h, img_w, img_h
        )
        root["images"].append(
            {
                "path_img": rel,
                "img_w": int(img_w),
                "img_h": int(img_h),
                "detections": dets,
            }
        )
        if (i + 1) % 50 == 0 or i + 1 == len(rels):
            print(f"  progress: {i+1}/{len(rels)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(root, indent=2) + "\n", encoding="utf-8")
    print(f"done: written={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
