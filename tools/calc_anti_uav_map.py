#!/usr/bin/env python3
"""Evaluate Anti-UAV300 tracker results at IoU 0.5 and 0.5:0.95.

The OSTrack result format contains one ``x,y,w,h`` box per frame and no
confidence score.  Anti-UAV is a single-object tracking dataset, so this
script uses the usual tracking adaptation of mAP: at an IoU threshold, a
target-present frame is a true positive when its predicted box reaches the
threshold; otherwise it is a miss.  ``mAP@0.5:0.95`` is the mean of these
frame-level AP/success values for IoU thresholds 0.50, 0.55, ..., 0.95.

Frames annotated with ``exist == 0`` are ignored, because a tracker result
does not include an objectness/confidence score with which false detections
could be ranked.

Examples:
  python tools/calc_anti_uav_map.py
  python tools/calc_anti_uav_map.py --per-sequence
  python tools/calc_anti_uav_map.py --results-dir /path/to/results --data-root /path/to/test-dev
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_DATA_ROOT = Path("/home/chenxinyi/data/anti_uav300/dataset/test-dev")
DEFAULT_RESULTS_DIR = Path(
    "/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300_uav300_wsp_A4"
)
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def is_valid_box(box: Optional[Sequence[float]]) -> bool:
    return box is not None and len(box) >= 4 and float(box[2]) > 0 and float(box[3]) > 0


def iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """IoU for boxes in x, y, width, height format."""
    if not is_valid_box(box1) or not is_valid_box(box2):
        return 0.0
    left, top = max(float(box1[0]), float(box2[0])), max(float(box1[1]), float(box2[1]))
    right = min(float(box1[0]) + float(box1[2]), float(box2[0]) + float(box2[2]))
    bottom = min(float(box1[1]) + float(box1[3]), float(box2[1]) + float(box2[3]))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = float(box1[2]) * float(box1[3]) + float(box2[2]) * float(box2[3]) - inter
    return inter / union if union > 0.0 else 0.0


def read_boxes(path: Path) -> list[list[float]]:
    boxes = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.strip().replace(",", " ").split()
            if not fields:
                continue
            if len(fields) < 4:
                raise ValueError(f"{path}:{number}: expected x y w h")
            try:
                boxes.append([float(v) for v in fields[:4]])
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: non-numeric bounding box") from exc
    return boxes


def target_exists(gt_box: object, exists: Sequence[object], frame: int) -> bool:
    if not is_valid_box(gt_box):
        return False
    if frame >= len(exists):
        return True
    try:
        return int(exists[frame]) == 1
    except (TypeError, ValueError):
        return bool(exists[frame])


def evaluate_sequence(label_path: Path, result_path: Path) -> tuple[int, list[int], int]:
    label = json.loads(label_path.read_text(encoding="utf-8"))
    gt_rects = label.get("gt_rect")
    if not isinstance(gt_rects, list):
        raise ValueError(f"{label_path}: missing list field 'gt_rect'")
    exists = label.get("exist", [1] * len(gt_rects))
    predictions = read_boxes(result_path)
    hits = [0] * len(IOU_THRESHOLDS)
    positives = 0
    for frame, gt_box in enumerate(gt_rects):
        if not target_exists(gt_box, exists, frame):
            continue
        positives += 1
        pred_box = predictions[frame] if frame < len(predictions) else None
        overlap = iou(gt_box, pred_box) if pred_box is not None else 0.0
        for index, threshold in enumerate(IOU_THRESHOLDS):
            hits[index] += overlap >= threshold
    return positives, hits, len(predictions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Anti-UAV300 test-dev root")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="directory containing *_IR.txt files")
    parser.add_argument("--modality", default="IR", choices=("IR", "RGB"), help="annotation/result modality")
    parser.add_argument("--per-sequence", action="store_true", help="also print scores for every sequence")
    args = parser.parse_args()

    label_paths = sorted(args.data_root.glob(f"*/{args.modality}_label.json"))
    if not label_paths:
        print(f"ERROR: no {args.modality}_label.json files under {args.data_root}", file=sys.stderr)
        return 1
    if not args.results_dir.is_dir():
        print(f"ERROR: results directory does not exist: {args.results_dir}", file=sys.stderr)
        return 1

    all_hits = [0] * len(IOU_THRESHOLDS)
    total_positives = 0
    missing, length_mismatches = [], []
    sequence_rows = []
    for label_path in label_paths:
        sequence = label_path.parent.name
        result_path = args.results_dir / f"{sequence}_{args.modality}.txt"
        if not result_path.is_file():
            missing.append(sequence)
            continue
        try:
            positives, hits, pred_count = evaluate_sequence(label_path, result_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        gt_frames = len(json.loads(label_path.read_text(encoding="utf-8"))["gt_rect"])
        if pred_count != gt_frames:
            length_mismatches.append(f"{sequence} (GT {gt_frames}, pred {pred_count})")
        total_positives += positives
        all_hits = [a + b for a, b in zip(all_hits, hits)]
        if args.per_sequence:
            rates = [hit / positives if positives else 0.0 for hit in hits]
            sequence_rows.append((sequence, positives, rates[0], sum(rates) / len(rates)))

    if not total_positives:
        print("ERROR: no target-present frames found", file=sys.stderr)
        return 1
    scores = [hit / total_positives for hit in all_hits]
    print(f"Sequences evaluated: {len(label_paths) - len(missing)}/{len(label_paths)}")
    print(f"Target-present frames: {total_positives:,}")
    print(f"mAP@0.5:      {scores[0] * 100:.2f}%")
    print(f"mAP@0.5:0.95: {sum(scores) / len(scores) * 100:.2f}%")
    print("Per-IoU AP: " + ", ".join(f"{t:.2f}={s * 100:.2f}%" for t, s in zip(IOU_THRESHOLDS, scores)))
    if args.per_sequence:
        print("\nsequence\tpositive_frames\tmAP@0.5\tmAP@0.5:0.95")
        for sequence, positives, map50, map5095 in sequence_rows:
            print(f"{sequence}\t{positives}\t{map50 * 100:.2f}%\t{map5095 * 100:.2f}%")
    if missing:
        print(f"WARNING: missing results for {len(missing)} sequences: {', '.join(missing[:10])}", file=sys.stderr)
    if length_mismatches:
        print("WARNING: result/GT length mismatches: " + ", ".join(length_mismatches[:10]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
