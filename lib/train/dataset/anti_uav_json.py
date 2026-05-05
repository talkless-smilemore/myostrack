"""
Anti-UAV410 / Anti-UAV600 等：与评测 `lib/test/evaluation/anti_uavdataset.py` 相同的目录与标注约定。
每场景目录下含 IR_label.json（或 RGB_label.json），gt_rect 为 [x,y,w,h]，可选 exist 表示可见。
帧图在 IR/、infrared/ 或场景根目录平铺。
"""
import os
from collections import OrderedDict

import numpy as np
import torch
import cv2

from lib.train.admin import env_settings
from lib.test.evaluation import anti_uavdataset as _anti_io
from .base_video_dataset import BaseVideoDataset


def _root_is_flat_anti_uav(root):
    try:
        names = os.listdir(root)
    except OSError:
        return False
    if not any(f.endswith("_label.json") for f in names):
        return False
    return bool(_anti_io._list_images(root))


class AntiUAVJson(BaseVideoDataset):
    """单模态（默认 IR）序列，用于 OSTrack 训练采样。"""

    def __init__(self, root=None, split="train", modalities="ir", image_loader=None):
        from lib.train.data import jpeg4py_loader

        if image_loader is None:
            image_loader = jpeg4py_loader
        if root is None:
            root = getattr(env_settings(), "anti_uav600_dir", "") or ""
        if not root:
            raise ValueError("请在 lib/train/admin/local.py 中设置 anti_uav600_dir，或在构造 AntiUAVJson(root=...) 时传入路径")
        scan_root = self._resolve_scan_root(root, split)
        super().__init__("AntiUAVJson", scan_root, image_loader)

        m = modalities.lower()
        if m not in ("ir", "rgb", "both"):
            raise ValueError("modalities 应为 ir / rgb / both")
        self._modalities = m

        self._entries = self._scan_entries(scan_root)
        self.sequence_list = list(range(len(self._entries)))
        self.class_list = ["uav"]
        self._seq_per_class = {"uav": list(self.sequence_list)}

    @staticmethod
    def _resolve_scan_root(root, split):
        if split == "train":
            for name in ("train", "training"):
                sub = os.path.join(root, name)
                if os.path.isdir(sub):
                    return sub
            return root
        if split == "val":
            for name in ("val", "validation", "valid"):
                sub = os.path.join(root, name)
                if os.path.isdir(sub):
                    return sub
            tried = ", ".join(os.path.join(root, n) for n in ("val", "validation", "valid"))
            raise ValueError(
                f"Anti-UAV: 未找到验证集目录（已尝试: {tried}）。"
                "请将 anti_uav600_dir 指到含 train 与 val/validation 子目录的父目录，"
                "或仅使用 ANTI_UAV600_TRAIN 并自定义 DATA.VAL。"
            )
        return root

    def _scan_entries(self, base_path):
        entries = []
        if not os.path.isdir(base_path):
            return entries

        child_scenes = []
        for name in sorted(os.listdir(base_path)):
            scene_dir = os.path.join(base_path, name)
            if os.path.isdir(scene_dir):
                child_scenes.append((name, scene_dir))

        for name, scene_dir in child_scenes:
            entries.extend(self._entries_from_scene_dir(scene_dir, name))

        if not entries and _root_is_flat_anti_uav(base_path):
            root_name = os.path.basename(os.path.normpath(base_path)) or "anti_uav"
            entries.extend(self._entries_from_scene_dir(base_path, root_name))

        return entries

    def _entries_from_scene_dir(self, scene_dir, scene_name):
        out = []
        try:
            fns = sorted(os.listdir(scene_dir))
        except OSError:
            return out

        for fn in fns:
            fn_low = fn.lower()
            label_stem = None
            if fn.endswith("_label.json"):
                label_stem = fn[: -len("_label.json")]
            elif fn_low in ("infrared.json", "visible.json", "ir.json", "rgb.json"):
                # Anti-UAV (video) commonly uses infrared.json / visible.json.
                if "infrared" in fn_low or fn_low == "ir.json":
                    label_stem = "IR"
                elif "visible" in fn_low or fn_low == "rgb.json":
                    label_stem = "RGB"
            if label_stem is None:
                continue

            u = label_stem.upper()
            if self._modalities == "ir" and u != "IR":
                continue
            if self._modalities == "rgb" and u != "RGB":
                continue

            json_path = os.path.join(scene_dir, fn)
            frames = _anti_io._resolve_frames(scene_dir, label_stem)
            video_path, video_len = self._resolve_video(scene_dir, u)
            if not frames and (video_path is None or video_len <= 0):
                continue

            try:
                gt_rect, exist = _anti_io._load_json_label(json_path)
            except (OSError, KeyError, ValueError):
                continue

            frame_num = len(frames) if frames else video_len
            gt = _anti_io._build_gt_array(gt_rect, exist, frame_num)
            start = _anti_io._first_positive_frame(gt)
            if start is None:
                continue

            gt = gt[start:, :]
            n = gt.shape[0]
            bbox = torch.from_numpy(gt.astype(np.float32))

            valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
            if exist is not None:
                vis_list = []
                for i in range(n):
                    if i + start < len(exist):
                        try:
                            vis_list.append(int(exist[i + start]) == 1)
                        except (TypeError, ValueError):
                            vis_list.append(bool(valid[i].item()))
                    else:
                        vis_list.append(bool(valid[i].item()))
                visible = torch.ByteTensor(vis_list) & valid.byte()
            else:
                visible = valid.byte()

            visible_ratio = visible.float()

            out.append(
                {
                    "name": f"{scene_name}_{label_stem}",
                    "frames": frames[start:] if frames else None,
                    "video_path": video_path if not frames else None,
                    "video_start": start,
                    "bbox": bbox,
                    "valid": valid,
                    "visible": visible,
                    "visible_ratio": visible_ratio,
                }
            )
        return out

    @staticmethod
    def _resolve_video(scene_dir, modality):
        candidates = []
        if modality == "IR":
            candidates = ["infrared.mp4", "IR.mp4", "ir.mp4"]
        elif modality == "RGB":
            candidates = ["visible.mp4", "RGB.mp4", "rgb.mp4"]

        for name in candidates:
            p = os.path.join(scene_dir, name)
            if not os.path.isfile(p):
                continue
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                cap.release()
                continue
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if length > 0:
                return p, length
        return None, 0

    def get_name(self):
        return "anti_uav_json"

    def has_class_info(self):
        return True

    def get_sequences_in_class(self, class_name):
        return self._seq_per_class[class_name]

    def get_sequence_info(self, seq_id):
        e = self._entries[seq_id]
        return {
            "bbox": e["bbox"],
            "valid": e["valid"],
            "visible": e["visible"],
            "visible_ratio": e["visible_ratio"],
        }

    def get_frames(self, seq_id, frame_ids, anno=None):
        e = self._entries[seq_id]
        if e.get("frames") is not None:
            frame_paths = e["frames"]
            frame_list = [self.image_loader(frame_paths[f_id]) for f_id in frame_ids]
        else:
            frame_list = self._load_video_frames(
                e["video_path"], frame_ids, start_idx=e.get("video_start", 0)
            )

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        obj_meta = OrderedDict(
            {
                "object_class_name": "uav",
                "motion_class": None,
                "major_class": None,
                "root_class": None,
                "motion_adverb": None,
            }
        )
        return frame_list, anno_frames, obj_meta

    @staticmethod
    def _load_video_frames(video_path, frame_ids, start_idx=0):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_list = []
        for fid in frame_ids:
            pos = int(start_idx + fid)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok or frame is None:
                # Fallback: return a black frame when decoding misses a frame.
                frame = np.zeros((512, 640, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
        cap.release()
        return frame_list

    def get_class_name(self, seq_id):
        return "uav"
