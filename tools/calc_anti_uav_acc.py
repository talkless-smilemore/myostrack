#!/usr/bin/env python3
"""Calculate frame-level ACC for Anti-UAV, Anti-UAV300 and Anti-UAV410.

A frame is correct when the predicted box has IoU >= the selected threshold
with the ground-truth box.  By default only frames with an existing target are
included.  Use --include-absent to include every annotated frame; an absent
target is correct only if the corresponding prediction is an empty box.

The default paths match this workspace.  For example:

    python tools/calc_anti_uav_acc.py
    python tools/calc_anti_uav_acc.py --dataset anti-uav410 --iou-threshold 0.5
    python tools/calc_anti_uav_acc.py --include-absent
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: str
    label_filename: str
    result_root: str


DEFAULT_SPECS = {
    "anti-uav": DatasetSpec(
        "anti-uav",
        "/home/chenxinyi/data/anti_uav/test",
        "infrared.json",
        "/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/"
        "vitb_256_mae_ce_32x4_ep300_uav_wsp_A4",
    ),
    "anti-uav300": DatasetSpec(
        "anti-uav300",
        "/home/chenxinyi/data/anti_uav300/dataset/test-dev",
        "IR_label.json",
        "/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/"
        "vitb_256_mae_ce_32x4_ep300_uav300_wsp_A4",
    ),
    "anti-uav410": DatasetSpec(
        "anti-uav410",
        "/home/chenxinyi/data/anti_uav410/test",
        "IR_label.json",
        "/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/"
        "vitb_256_mae_ce_32x4_ep300_uav410_wsp_A4",
    ),
}


@dataclass
class Score:
    name: str
    sequences: int
    correct: int
    total: int
    missing_results: List[str]
    length_mismatches: List[Tuple[str, int, int]]

    @property
    def acc(self) -> float:
        return self.correct / self.total if self.total else 0.0


def read_predictions(path: str) -> List[List[float]]:
    """Read OSTrack result lines in either tab-, space-, or comma-separated form."""
    boxes = []
    with open(path, encoding="utf-8") as result_file:
        for line_number, line in enumerate(result_file, start=1):
            values = line.strip().replace(",", " ").split()
            if not values:
                continue
            if len(values) < 4:
                raise ValueError(f"{path}:{line_number}: expected at least four values")
            try:
                boxes.append([float(value) for value in values[:4]])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric bounding box") from exc
    return boxes


def valid_box(box: Optional[Sequence[float]]) -> bool:
    return box is not None and len(box) >= 4 and float(box[2]) > 0 and float(box[3]) > 0


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    if not valid_box(a) or not valid_box(b):
        return 0.0
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[0]) + float(a[2]), float(b[0]) + float(b[2]))
    bottom = min(float(a[1]) + float(a[3]), float(b[1]) + float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = float(a[2]) * float(a[3]) + float(b[2]) * float(b[3]) - intersection
    return intersection / union if union > 0 else 0.0


def target_exists(gt_box: Optional[Sequence[float]], exist: Sequence[object], frame: int) -> bool:
    if frame >= len(exist):
        return valid_box(gt_box)
    try:
        return int(exist[frame]) == 1 and valid_box(gt_box)
    except (TypeError, ValueError):
        return valid_box(gt_box)


def score_dataset(spec: DatasetSpec, threshold: float, include_absent: bool) -> Score:
    label_paths = sorted(glob.glob(os.path.join(spec.root, "*", spec.label_filename)))
    if not label_paths:
        raise FileNotFoundError(f"No {spec.label_filename} files found below {spec.root}")

    score = Score(spec.name, 0, 0, 0, [], [])
    for label_path in label_paths:
        scene = os.path.basename(os.path.dirname(label_path))
        result_path = os.path.join(spec.result_root, f"{scene}_IR.txt")
        if not os.path.isfile(result_path):
            score.missing_results.append(scene)
            continue

        with open(label_path, encoding="utf-8") as label_file:
            label = json.load(label_file)
        ground_truth = label.get("gt_rect")
        if ground_truth is None:
            raise KeyError(f"{label_path} has no gt_rect field")
        exists = label.get("exist", [1] * len(ground_truth))
        predictions = read_predictions(result_path)
        score.sequences += 1
        if len(predictions) != len(ground_truth):
            score.length_mismatches.append((scene, len(ground_truth), len(predictions)))

        for frame, gt_box in enumerate(ground_truth):
            gt_exists = target_exists(gt_box, exists, frame)
            prediction = predictions[frame] if frame < len(predictions) else None
            if not gt_exists and not include_absent:
                continue
            score.total += 1
            if gt_exists:
                if prediction is not None and box_iou(gt_box, prediction) >= threshold:
                    score.correct += 1
            elif not valid_box(prediction):
                score.correct += 1
    return score


def print_scores(scores: Sequence[Score]) -> None:
    print(f"{'dataset':<15}{'sequences':>10}{'correct/total':>20}{'ACC':>12}")
    print("-" * 57)
    for score in scores:
        print(f"{score.name:<15}{score.sequences:>10}{score.correct:>9,}/{score.total:<10,}{score.acc:>11.2%}")
    if len(scores) > 1:
        correct = sum(score.correct for score in scores)
        total = sum(score.total for score in scores)
        print("-" * 57)
        print(f"{'combined':<15}{sum(s.sequences for s in scores):>10}{correct:>9,}/{total:<10,}{correct / total:>11.2%}")
        print(f"{'dataset mean':<15}{'':>10}{'':>20}{sum(s.acc for s in scores) / len(scores):>11.2%}")
    for score in scores:
        if score.missing_results:
            print(f"WARNING [{score.name}]: missing result files for {len(score.missing_results)} sequences: "
                  + ", ".join(score.missing_results[:10]))
        if score.length_mismatches:
            details = ", ".join(f"{name} (GT {gt}, pred {pred})" for name, gt, pred in score.length_mismatches[:5])
            print(f"WARNING [{score.name}]: {len(score.length_mismatches)} result/GT length mismatches: {details}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["all", *DEFAULT_SPECS], default="all",
                        help="dataset to score (default: all)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU required for a target-present frame to be correct (default: 0.5)")
    parser.add_argument("--include-absent", action="store_true",
                        help="include frames where the annotation says the target is absent")
    args = parser.parse_args()
    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be in [0, 1]")

    specs = list(DEFAULT_SPECS.values()) if args.dataset == "all" else [DEFAULT_SPECS[args.dataset]]
    try:
        scores = [score_dataset(spec, args.iou_threshold, args.include_absent) for spec in specs]
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"IoU threshold: {args.iou_threshold:g}; frames: {'all annotated' if args.include_absent else 'target-present only'}")
    print_scores(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
