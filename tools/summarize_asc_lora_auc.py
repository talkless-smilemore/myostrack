#!/usr/bin/env python3
"""Collect AUC from the standard OSTrack eval_data.pkl files into a CSV."""
from __future__ import annotations
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


EXPERIMENTS = ("F0_full", "F1_no_channel_gate", "F2_no_spectral_complement",
               "F3_hard_spectral_complement", "F4_no_complement_constraint", "F5_standard_lora")
DATASETS = ("anti_uav", "anti_uav300", "anti_uav410")


def auc_from_eval_data(path: Path) -> float:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    valid = np.asarray(data["valid_sequence"], dtype=bool)
    curve = np.asarray(data["ave_success_rate_plot_overlap"], dtype=float)
    # [trackers, sequences, thresholds] in the normal single-run protocol.
    if curve.ndim != 3 or curve.shape[0] != 1:
        raise ValueError(f"Expected one tracker in {path}, got curve shape {curve.shape}")
    return float(curve[0, valid, :].mean(axis=0).mean() * 100.0)


def locate(root: Path, experiment: str, dataset: str) -> Path | None:
    matches = list(root.glob(f"**/*{experiment}*/**/{dataset}/eval_data.pkl"))
    matches += list(root.glob(f"**/{dataset}/**/{experiment}/eval_data.pkl"))
    # Keep the script usable with the stock plot folder naming as well.
    matches += list(root.glob(f"**/*{experiment}*_{dataset}/eval_data.pkl"))
    return sorted(set(matches))[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Directory containing per-experiment/dataset eval_data.pkl files")
    parser.add_argument("--output", type=Path, default=Path("output/asc_lora_ablations/auc_summary.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("experiment", "Anti-UAV", "Anti-UAV300", "Anti-UAV410"))
        writer.writeheader()
        for experiment in EXPERIMENTS:
            row = {"experiment": experiment}
            for dataset, label in zip(DATASETS, writer.fieldnames[1:]):
                path = locate(args.results_root, experiment, dataset)
                row[label] = "" if path is None else f"{auc_from_eval_data(path):.4f}"
            writer.writerow(row)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
