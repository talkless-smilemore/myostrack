"""
Analyse frame-to-frame target motion in Anti-UAV datasets.

Computes:
 - target size distribution (sqrt(w*h))
 - centre displacement between consecutive visible frames
 - the search factor required to keep the target within the crop

Usage:
    python tools/analyze_anti_uav_motion.py

Output:
    - prints summary tables to stdout
    - saves a JSON file to tools/anti_uav_motion_stats.json
"""

import json
import math
import os
import re
import sys
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Paths — match the directories configured in lib/train/admin/local.py
# ---------------------------------------------------------------------------
DATASETS = {
    "anti_uav":    r"F:\uav_database\anti_uav",
    "anti_uav300": r"F:\uav_database\anti_uav300",
    "anti_uav410": r"F:\uav_database\anti_uav410",
    "anti_uav600": r"F:\uav_database\anti_uav600\3rd_Anti-UAV_train_val",
}

OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "anti_uav_motion_stats.json")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_images(folder):
    try:
        return sorted(
            [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in _IMG_EXTS],
            key=lambda p: [int(t) if t.isdigit() else t.lower()
                           for t in re.split(r"(\d+)", os.path.basename(p))]
        )
    except OSError:
        return []


def _label_stem(filename):
    fn_low = filename.lower()
    if fn_low.endswith("_label.json"):
        return filename[:-len("_label.json")]
    if fn_low in ("infrared.json", "visible.json", "ir.json", "rgb.json"):
        return "IR" if fn_low in ("infrared.json", "ir.json") else "RGB"
    return None


def _resolve_frames(scene_dir, label_stem):
    """Try IR/ / RGB/ sub-dir first, then scene root."""
    for sub in [label_stem, label_stem.upper(), label_stem.lower(),
                "infrared" if label_stem.upper() == "IR" else None,
                "visible" if label_stem.upper() == "RGB" else None]:
        if sub is None:
            continue
        frames = _list_images(os.path.join(scene_dir, sub))
        if frames:
            return frames
    return _list_images(scene_dir)


def _scan_scene(scene_dir):
    """Return list of (name, gt_array) for each label in a scene directory."""
    results = []
    try:
        fns = sorted(os.listdir(scene_dir))
    except OSError:
        return results

    for fn in fns:
        stem = _label_stem(fn)
        if stem is None:
            continue

        json_path = os.path.join(scene_dir, fn)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        gt_rect = data.get("gt_rect")
        exist = data.get("exist")
        if gt_rect is None:
            continue

        frames = _resolve_frames(scene_dir, stem)
        n_frames = len(frames) if frames else max(len(gt_rect), 1)

        # build gt (n_frames, 4)
        arr = np.zeros((n_frames, 4), dtype=np.float64)
        L = min(len(gt_rect), n_frames)
        for i in range(L):
            box = gt_rect[i]
            if box is None or len(box) < 4:
                continue
            x, y, w, h = [float(box[j]) for j in range(4)]
            if exist is not None and i < len(exist):
                try:
                    if int(exist[i]) != 1:
                        continue
                except Exception:
                    pass
            if w > 0 and h > 0:
                arr[i] = [x, y, w, h]

        # clip invalid leading frames
        start = 0
        for i in range(arr.shape[0]):
            if arr[i, 2] > 0 and arr[i, 3] > 0:
                start = i
                break
        arr = arr[start:]

        if arr.shape[0] < 2:
            continue

        seq_name = f"{os.path.basename(scene_dir)}_{stem}"
        results.append((seq_name, arr))

    return results


def _scan_dataset(root):
    """Walk scene sub-directories under root."""
    all_seqs = []
    if not os.path.isdir(root):
        return all_seqs

    has_subdirs = False
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d):
            has_subdirs = True
            all_seqs.extend(_scan_scene(d))

    # flat layout fallback
    if not has_subdirs and not all_seqs:
        all_seqs.extend(_scan_scene(root))

    return all_seqs


def required_search_factor(target_w, target_h, displacement):
    """Search factor so that crop half-side covers the displacement.

      crop_side = sqrt(w*h) * sf
    → half_side = sqrt(w*h) * sf / 2
    → sf        = 2 * displacement / sqrt(w*h)
    """
    diag = max(1e-3, math.sqrt(max(1.0, target_w) * max(1.0, target_h)))
    return 2.0 * displacement / diag


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def analyse():
    all_results = {}   # ds_name → stats dict

    for ds_name, ds_root in DATASETS.items():
        if not os.path.isdir(ds_root):
            print(f"[SKIP] {ds_name}: {ds_root}  not found")
            continue

        print(f"\n{'='*70}")
        print(f" Dataset: {ds_name}")
        print(f" Path:    {ds_root}")

        seqs = _scan_dataset(ds_root)

        # anti_uav600 may need train/ sub-directory
        if not seqs:
            for sub in ("train", "training"):
                subpath = os.path.join(ds_root, sub)
                if os.path.isdir(subpath):
                    seqs = _scan_dataset(subpath)
                    if seqs:
                        print(f"  (read from {sub}/ sub-directory)")
                        break

        print(f" Sequences found: {len(seqs)}")
        if not seqs:
            continue

        sizes = []       # sqrt(w*h) per valid frame
        disps = []       # displacement between consecutive valid frames
        req_sf = []      # required search factor per transition
        seq_stats = []

        for seq_name, gt in seqs:
            centres = np.column_stack([
                gt[:, 0] + gt[:, 2] / 2.0,
                gt[:, 1] + gt[:, 3] / 2.0,
            ])
            diags = np.sqrt(np.maximum(gt[:, 2] * gt[:, 3], 1.0))
            valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
            v_centres = centres[valid]
            v_diags = diags[valid]
            v_gt = gt[valid]
            nv = len(v_centres)

            if nv < 2:
                continue

            sizes.extend(v_diags.tolist())

            d = np.linalg.norm(v_centres[1:] - v_centres[:-1], axis=1)
            disps.extend(d.tolist())

            seq_req = []
            for i in range(len(d)):
                r = required_search_factor(v_gt[i, 2], v_gt[i, 3], d[i])
                req_sf.append(r)
                seq_req.append(r)

            seq_req = np.array(seq_req) if seq_req else np.array([])
            seq_stats.append({
                "name": seq_name,
                "frames": int(gt.shape[0]),
                "valid_frames": int(valid.sum()),
                "mean_size_diag": float(np.mean(v_diags)),
                "median_disp": float(np.median(d)) if len(d) else 0.0,
                "max_disp": float(np.max(d)) if len(d) else 0.0,
                "p90_disp": float(np.percentile(d, 90)) if len(d) else 0.0,
                "p95_disp": float(np.percentile(d, 95)) if len(d) else 0.0,
                "p99_disp": float(np.percentile(d, 99)) if len(d) else 0.0,
                "p95_req_sf": float(np.percentile(seq_req, 95)) if len(seq_req) else 0.0,
            })

        if not sizes:
            print("  No valid frames.")
            continue

        sizes = np.array(sizes)
        disps = np.array(disps)
        req_sf = np.array(req_sf)

        # ---- print stats ----
        print(f"\n  Valid transitions: {len(disps):,}")

        print(f"\n  {'─'*50}")
        print(f"  Target size  sqrt(w*h) (pixels)")
        print(f"  {'─'*50}")
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"    P{p:2d}:  {np.percentile(sizes, p):6.1f} px")

        print(f"\n  {'─'*50}")
        print(f"  Frame-to-frame centre displacement (pixels)")
        print(f"  {'─'*50}")
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"    P{p:2d}:  {np.percentile(disps, p):6.1f} px")

        print(f"\n  {'─'*50}")
        print(f"  Required search factor  (2*disp / sqrt(w*h))")
        print(f"  {'─'*50}")
        for p in [5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]:
            print(f"    P{p:4.1f}:  {np.percentile(req_sf, p):6.2f}")

        print(f"\n  {'─'*50}")
        print(f"  Coverage by search factor")
        print(f"  {'─'*50}")
        for sf in [2.0, 2.5, 3.0, 3.5, 3.6, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0]:
            cov = (req_sf <= sf).mean() * 100
            bar = "█" * int(cov / 5)
            print(f"    sf ≤ {sf:4.1f}:  {cov:5.1f}%  {bar}")

        print(f"\n  {'─'*50}")
        print(f"  Displacement coverage")
        print(f"  {'─'*50}")
        for thr in [2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150]:
            cov = (disps <= thr).mean() * 100
            print(f"    disp ≤ {thr:3d} px:  {cov:5.1f}%")

        # top hardest sequences
        seq_stats.sort(key=lambda s: s["p95_req_sf"], reverse=True)
        print(f"\n  {'─'*50}")
        print(f"  Top 10 hardest sequences  (by P95 required sf)")
        print(f"  {'─'*50}")
        for s in seq_stats[:10]:
            print(f"    {s['name']:<55s} valid={s['valid_frames']:5d}"
                  f"  sz={s['mean_size_diag']:5.1f}px"
                  f"  P95_disp={s['p95_disp']:5.1f}px"
                  f"  P95_sf={s['p95_req_sf']:5.2f}")

        # ---- save for JSON ----
        all_results[ds_name] = {
            "n_sequences": len(seqs),
            "n_transitions": len(disps),
            "size_percentiles": {f"P{p}": float(np.percentile(sizes, p))
                                 for p in [5, 10, 25, 50, 75, 90, 95, 99]},
            "disp_percentiles": {f"P{p}": float(np.percentile(disps, p))
                                 for p in [5, 10, 25, 50, 75, 90, 95, 99]},
            "req_sf_percentiles": {f"P{p}": float(np.percentile(req_sf, p))
                                   for p in [5, 10, 25, 50, 75, 90, 95, 99]},
            "sf_coverage": {f"sf_{sf}": float((req_sf <= sf).mean())
                            for sf in [2.0, 2.5, 3.0, 3.5, 3.6, 4.0, 4.5,
                                       5.0, 5.5, 6.0, 7.0, 8.0]},
            "top10_hard_seq": seq_stats[:10],
        }

    # ---- aggregate across datasets ----
    if all_results:
        print(f"\n{'='*70}")
        print(f" COMBINED  (all datasets)")
        print(f"{'='*70}")
        all_sf = []
        for ds in all_results.values():
            all_sf.extend([ds["req_sf_percentiles"][k]
                           for k in ["P50", "P90", "P95", "P99"]])
        # Count sequences total
        total_seqs = sum(d["n_sequences"] for d in all_results.values())
        total_trans = sum(d["n_transitions"] for d in all_results.values())
        print(f" Total sequences:    {total_seqs}")
        print(f" Total transitions:  {total_trans:,}")

        print(f"\n  {'─'*50}")
        print(f"  Recommended adaptive search factor ranges")
        print(f"  {'─'*50}")
        for ds_name, ds_data in all_results.items():
            p90 = ds_data["req_sf_percentiles"]["P90"]
            p95 = ds_data["req_sf_percentiles"]["P95"]
            p99 = ds_data["req_sf_percentiles"]["P99"]
            p50_disp = ds_data["disp_percentiles"]["P50"]
            p95_disp = ds_data["disp_percentiles"]["P95"]
            p50_sz = ds_data["size_percentiles"]["P50"]
            print(f"\n  {ds_name}:")
            print(f"    median size:       {p50_sz:.1f} px")
            print(f"    median disp:       {p50_disp:.1f} px/frame")
            print(f"    P95 disp:          {p95_disp:.1f} px/frame")
            print(f"    P90 req sf:        {p90:.2f}")
            print(f"    P95 req sf:        {p95:.2f}")
            print(f"    P99 req sf:        {p99:.2f}")
            print(f"    → sf_base 建议:    {max(3.0, p90):.1f}  (cover 90%)")
            print(f"    → sf_max 建议:     {max(4.0, p95):.1f}  (cover 95%)")
            print(f"    → sf_upper 建议:   {max(5.0, p99):.1f}  (cover 99%)")

    # ---- write JSON ----
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    analyse()
