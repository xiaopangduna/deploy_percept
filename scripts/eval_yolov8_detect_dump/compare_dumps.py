#!/usr/bin/env python3
"""Compare board vs ONNX detection dump JSON (after optional score filter)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare detect dump JSONs")
    p.add_argument("--board", required=True, help="Board dump JSON")
    p.add_argument("--onnx", required=True, help="ONNX dump JSON")
    p.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Keep detections with score >= T (must be >= dump_conf)",
    )
    p.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for matching same class_id",
    )
    return p.parse_args()


def load_dump(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def filter_dets(dets: list[dict[str, Any]], score_thresh: float) -> list[dict[str, Any]]:
    out = [d for d in dets if float(d["score"]) >= score_thresh]
    out.sort(key=lambda d: float(d["score"]), reverse=True)
    return out


def iou_xyxy(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
    bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def match_image(
    board_dets: list[dict[str, Any]],
    onnx_dets: list[dict[str, Any]],
    iou_thresh: float,
) -> dict[str, Any]:
    """Greedy match by score: same class_id, IoU >= thresh."""
    used_onnx: set[int] = set()
    pairs: list[dict[str, Any]] = []
    board_only = 0
    for bd in board_dets:
        best_j = -1
        best_iou = 0.0
        for j, od in enumerate(onnx_dets):
            if j in used_onnx:
                continue
            if int(bd["class_id"]) != int(od["class_id"]):
                continue
            iou = iou_xyxy(bd, od)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j < 0:
            board_only += 1
            continue
        used_onnx.add(best_j)
        od = onnx_dets[best_j]
        pairs.append(
            {
                "class_id": int(bd["class_id"]),
                "iou": best_iou,
                "score_board": float(bd["score"]),
                "score_onnx": float(od["score"]),
                "abs_score_delta": abs(float(bd["score"]) - float(od["score"])),
            }
        )
    onnx_only = len(onnx_dets) - len(used_onnx)
    return {
        "matched": len(pairs),
        "board_only": board_only,
        "onnx_only": onnx_only,
        "pairs": pairs,
    }


def check_meta(board: dict[str, Any], onnx: dict[str, Any], score_thresh: float) -> None:
    bm, om = board.get("meta", {}), onnx.get("meta", {})
    for key in ("dump_conf", "nms", "preprocess", "coord_space", "model_in_w", "model_in_h"):
        if bm.get(key) != om.get(key):
            print(
                f"warning: meta.{key} differs: board={bm.get(key)!r} onnx={om.get(key)!r}",
                file=sys.stderr,
            )
    dump_conf = float(bm.get("dump_conf", 0.0))
    if score_thresh + 1e-9 < dump_conf:
        print(
            f"warning: --score-thresh {score_thresh} < dump_conf {dump_conf}; "
            "boxes below dump_conf were never recorded",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()
    board = load_dump(Path(args.board))
    onnx = load_dump(Path(args.onnx))
    check_meta(board, onnx, args.score_thresh)

    board_map = {im["path_img"]: im for im in board.get("images", [])}
    onnx_map = {im["path_img"]: im for im in onnx.get("images", [])}
    keys = sorted(set(board_map) | set(onnx_map))

    total_matched = 0
    total_board_only = 0
    total_onnx_only = 0
    total_board = 0
    total_onnx = 0
    iou_sum = 0.0
    score_delta_sum = 0.0
    only_board_imgs = 0
    only_onnx_imgs = 0

    for path in keys:
        if path not in board_map:
            only_onnx_imgs += 1
            continue
        if path not in onnx_map:
            only_board_imgs += 1
            continue
        bd = filter_dets(board_map[path].get("detections", []), args.score_thresh)
        od = filter_dets(onnx_map[path].get("detections", []), args.score_thresh)
        total_board += len(bd)
        total_onnx += len(od)
        m = match_image(bd, od, args.iou_thresh)
        total_matched += m["matched"]
        total_board_only += m["board_only"]
        total_onnx_only += m["onnx_only"]
        for pair in m["pairs"]:
            iou_sum += pair["iou"]
            score_delta_sum += pair["abs_score_delta"]

    mean_iou = iou_sum / total_matched if total_matched else 0.0
    mean_abs_score_delta = score_delta_sum / total_matched if total_matched else 0.0
    match_rate_board = total_matched / total_board if total_board else 0.0
    match_rate_onnx = total_matched / total_onnx if total_onnx else 0.0

    print("compare_detect_dumps")
    print(f"  images_common     : {len(keys) - only_board_imgs - only_onnx_imgs}")
    print(f"  images_board_only : {only_board_imgs}")
    print(f"  images_onnx_only  : {only_onnx_imgs}")
    print(f"  score_thresh      : {args.score_thresh}")
    print(f"  iou_thresh        : {args.iou_thresh}")
    print(f"  dets_board        : {total_board}")
    print(f"  dets_onnx         : {total_onnx}")
    print(f"  matched           : {total_matched}")
    print(f"  board_only        : {total_board_only}")
    print(f"  onnx_only         : {total_onnx_only}")
    print(f"  match_rate_board  : {match_rate_board:.4f}")
    print(f"  match_rate_onnx   : {match_rate_onnx:.4f}")
    print(f"  mean_iou          : {mean_iou:.4f}")
    print(f"  mean_abs_dscore   : {mean_abs_score_delta:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
