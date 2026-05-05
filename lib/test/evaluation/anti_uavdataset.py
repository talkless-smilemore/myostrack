"""
Anti-UAV 系列数据集约定的单目标跟踪评测适配（与 OSTrack 的 Sequence 接口对齐）。

支持两种目录形式：

1) Anti-UAV410 常见：一个文件夹内平铺所有帧图 + 同目录下 IR_label.json（或 RGB_label.json）。
   可将 anti_uav_path 直接指向该文件夹；或指向「多个此类文件夹」的父目录。

2) 按模态分子目录：每场景下 IR/、RGB/（或 infrared/、visible/）放图片，旁路放 *_label.json。

标注 JSON 内含 gt_rect: 每帧 [x, y, w, h]；可选 exist（1 表示可见）。
"""
import json
import os
import re
import cv2

import numpy as np

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList


def _natural_sort_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    names = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    names.sort(key=_natural_sort_key)
    return [os.path.join(folder, f) for f in names]


def _resolve_image_dir(scene_dir, label_stem):
    """label_stem 例如 IR / RGB（来自 IR_label.json）。"""
    stem = label_stem.upper()
    candidates = [
        os.path.join(scene_dir, label_stem),
        os.path.join(scene_dir, stem),
        os.path.join(scene_dir, stem.lower()),
    ]
    if stem == "IR":
        candidates.extend(
            [
                os.path.join(scene_dir, "infrared"),
                os.path.join(scene_dir, "IR"),
            ]
        )
    elif stem == "RGB":
        candidates.extend(
            [
                os.path.join(scene_dir, "visible"),
                os.path.join(scene_dir, "RGB"),
            ]
        )
    for d in candidates:
        frames = _list_images(d)
        if frames:
            return d, frames
    return None, []


def _resolve_frames(scene_dir, label_stem):
    """先找 IR/、RGB 等子目录；没有则用 scene_dir 根目录下所有图片（Anti-UAV410 平铺）。"""
    _, frames = _resolve_image_dir(scene_dir, label_stem)
    if frames:
        return frames
    return _list_images(scene_dir)


def _resolve_video(scene_dir, label_stem):
    stem = label_stem.upper()
    candidates = []
    if stem == "IR":
        candidates = ["IR.mp4", "ir.mp4", "infrared.mp4"]
    elif stem == "RGB":
        candidates = ["RGB.mp4", "rgb.mp4", "visible.mp4"]

    for n in candidates:
        p = os.path.join(scene_dir, n)
        if os.path.isfile(p):
            return p
    return None


def _video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _load_json_label(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt_rect = data.get("gt_rect")
    if gt_rect is None:
        raise KeyError(f"{json_path} 缺少 gt_rect 字段")
    exist = data.get("exist")
    return gt_rect, exist


def _build_gt_array(gt_rect, exist, n_frames):
    """输出 (n_frames, 4)，与帧数对齐；无效帧用 0 占位。"""
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
            except (TypeError, ValueError):
                pass
        if w > 0 and h > 0 and (abs(x) + abs(y) + w + h) > 0:
            arr[i, :] = [x, y, w, h]
    return arr


def _first_positive_frame(gt):
    for i in range(gt.shape[0]):
        if gt[i, 2] > 0 and gt[i, 3] > 0:
            return i
    return None


class AntiUAVDataset(BaseDataset):
    """
    Args:
        split: 保留接口。
        modalities: 'ir' / 'rgb' / 'both'。Anti-UAV410 仅红外时用 'ir' 或数据集名 anti_uav410。
    """

    def __init__(self, split="test", modalities="both", env_attr="anti_uav_path"):
        super().__init__()
        self.base_path = getattr(self.env_settings, env_attr, "")
        self.split = split
        m = modalities.lower()
        if m not in ("ir", "rgb", "both"):
            raise ValueError("modalities 应为 ir / rgb / both")
        self.modalities = m
        self._entries = self._scan_dataset()

    def __len__(self):
        return len(self._entries)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(e) for e in self._entries])

    def _scan_dataset(self):
        entries = []
        if not self.base_path or not os.path.isdir(self.base_path):
            return entries

        child_scenes = []
        for name in sorted(os.listdir(self.base_path)):
            scene_dir = os.path.join(self.base_path, name)
            if os.path.isdir(scene_dir):
                child_scenes.append((name, scene_dir))

        for name, scene_dir in child_scenes:
            entries.extend(self._entries_from_scene_dir(scene_dir, name))

        # 根目录即单条序列：无子文件夹或子文件夹未解析出任何序列时，尝试平铺图 + *_label.json
        if not entries and self._root_is_flat_sequence(self.base_path):
            root_name = os.path.basename(os.path.normpath(self.base_path)) or "anti_uav410"
            entries.extend(self._entries_from_scene_dir(self.base_path, root_name))

        return entries

    def _root_is_flat_sequence(self, root):
        try:
            names = os.listdir(root)
        except OSError:
            return False
        if not any(f.endswith("_label.json") for f in names):
            return False
        return bool(_list_images(root))

    def _entries_from_scene_dir(self, scene_dir, scene_name):
        out = []
        try:
            fns = sorted(os.listdir(scene_dir))
        except OSError:
            return out

        for fn in fns:
            if not fn.endswith("_label.json"):
                continue
            label_stem = fn[: -len("_label.json")]
            u = label_stem.upper()
            if self.modalities == "ir" and u != "IR":
                continue
            if self.modalities == "rgb" and u != "RGB":
                continue

            json_path = os.path.join(scene_dir, fn)
            frames = _resolve_frames(scene_dir, label_stem)
            video_path = None
            if not frames:
                video_path = _resolve_video(scene_dir, label_stem)
                if video_path is None:
                    continue

            try:
                gt_rect, exist = _load_json_label(json_path)
            except (OSError, KeyError, json.JSONDecodeError):
                continue

            if frames:
                n_frames = len(frames)
            else:
                n_frames = _video_frame_count(video_path)
                if n_frames <= 0:
                    continue
                # Use a virtual frame URI so tracker can decode mp4 lazily.
                frames = [f"video://{video_path}#{i}" for i in range(n_frames)]

            gt = _build_gt_array(gt_rect, exist, len(frames))
            start = _first_positive_frame(gt)
            if start is None:
                continue

            seq_name = f"{scene_name}_{label_stem}"
            out.append(
                {
                    "name": seq_name,
                    "frames": frames,
                    "gt": gt,
                    "start": start,
                }
            )
        return out

    def _construct_sequence(self, entry):
        start = entry["start"]
        frames = entry["frames"][start:]
        gt = entry["gt"][start:, :]
        return Sequence(
            entry["name"],
            frames,
            "anti_uav",
            gt,
            object_class="aircraft",
        )
