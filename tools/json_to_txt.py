 #!/usr/bin/env python3
"""
将 Ground Truth JSON 转换为跟踪评估常用的 TXT 格式。

输出格式：每行一个目标框
    x,y,w,h

支持的常见 JSON 格式：

1. {"gt_rect": [[x, y, w, h], [x, y, w, h], ...]}
2. {"bboxes": [[x, y, w, h], ...]}
3. {"frames": [{"bbox": [x, y, w, h]}, ...]}
4. [[x, y, w, h], [x, y, w, h], ...]

用法示例：
    python tools/json_to_txt.py input.json output.txt

Anti-UAV 数据集示例：
    python tools/json_to_txt.py \
        /home/chenxinyi/data/anti_uav/test/20190925_111757_1_1/infrared.json \
        /home/chenxinyi/data/anti_uav/test/20190925_111757_1_1/groundtruth.txt

如果 JSON 中的字段名称不是上述名称，可以手动指定：
    python tools/json_to_txt.py input.json output.txt --key annotations
"""

import argparse
import json
from pathlib import Path
from typing import Any, List

# ======================== 请在这里修改 ========================
# 要转换的 JSON 文件
INPUT_JSON = "/home/chenxinyi/data/anti_uav/test/20190925_134301_1_9/infrared.json"

# 转换后保存的 TXT 文件
OUTPUT_TXT = "/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/groundtruth/20190925_134301_1_9_IR.txt"

# 如果 JSON 使用 gt_rect 字段，保持 None 即可；其他字段可填写字段名，例如 "annotations"
JSON_BOX_KEY = None
# ==============================================================


BOX_KEYS = ("bbox", "box", "rect", "ground_truth", "groundtruth", "gt")
LIST_KEYS = ("gt_rect", "gt_rects", "bboxes", "boxes", "annotations", "frames", "results")


def is_box(value: Any) -> bool:
    """判断一个对象是否为 [x, y, w, h]。"""
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 4
        and all(isinstance(item, (int, float)) for item in value[:4])
    )


def find_boxes(data: Any, preferred_key: str = None) -> List[List[float]]:
    """从常见 JSON 结构中递归提取边界框。"""
    if is_box(data):
        return [list(data[:4])]

    if isinstance(data, list):
        boxes = []
        for item in data:
            boxes.extend(find_boxes(item))
        return boxes

    if isinstance(data, dict):
        keys = ([preferred_key] if preferred_key else []) + list(LIST_KEYS) + list(BOX_KEYS)
        visited = set()
        for key in keys:
            if key in visited or key not in data:
                continue
            visited.add(key)
            boxes = find_boxes(data[key])
            if boxes:
                return boxes

        # 兼容字段名称不固定的嵌套结构，例如 data -> sequence -> frames
        for value in data.values():
            boxes = find_boxes(value)
            if boxes:
                return boxes

    return []


def convert(input_path: Path, output_path: Path, key: str = None) -> int:
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Anti-UAV 的标注格式是：{"exist": [...], "gt_rect": [[x,y,w,h], ...]}
    # 显式优先读取 gt_rect，避免递归读取到 exist 等其他数组。
    if key is None and isinstance(data, dict) and "gt_rect" in data:
        boxes = find_boxes(data["gt_rect"])
    else:
        boxes = find_boxes(data, preferred_key=key)
    if not boxes:
        raise ValueError(
            "没有找到边界框。请检查 JSON 格式，或使用 --key 指定保存边界框的字段。"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for box in boxes:
            file.write(",".join(format_number(value) for value in box) + "\n")

    return len(boxes)


def format_number(value: float) -> str:
    """整数不带小数，浮点数保留原有精度表现。"""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".10g")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Ground Truth JSON 转换为 x,y,w,h TXT")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path(INPUT_JSON),
        help="输入 JSON 文件（不填写时使用脚本顶部的 INPUT_JSON）",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path(OUTPUT_TXT),
        help="输出 TXT 文件（不填写时使用脚本顶部的 OUTPUT_TXT）",
    )
    parser.add_argument(
        "--key",
        default=JSON_BOX_KEY,
        help="边界框所在字段名，例如 gt_rect、annotations、frames",
    )
    args = parser.parse_args()

    count = convert(args.input, args.output, args.key)
    print(f"转换完成：{count} 个边界框")
    print(f"输出文件：{args.output}")


if __name__ == "__main__":
    main()
