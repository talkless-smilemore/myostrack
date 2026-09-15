import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os
import tqdm
from scipy.signal import savgol_filter
import matplotlib.ticker as ticker

# 图片输出目录（请使用自己有写权限的目录）
OUTPUT_DIR = "/home/chenxinyi/OSTrack-main/output/sglora/test/draw_IoU"

def load_text_numpy(path, delimiter=' ', dtype=np.float32):

    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                if 'AttriSeqsTxt' in path or  'TypeSeqsTxt' in path:
                    ground_truth_rect = np.loadtxt(path, delimiter=',', dtype=dtype)
                else:
                    ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype, skiprows=1)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def read_bbox_file(file_path):
    """
    从文件中读取边界框信息
    :param file_path: 文件路径
    :return: 边界框数组，每行包含左上角坐标 (x, y) 以及宽 (w) 和高 (h)
    """
    bboxes = load_text_numpy(file_path,delimiter=[',', ' ', '\t'])
    return bboxes


def calculate_iou(box1, box2):
    """
    计算两个边界框的 IoU
    :param box1: 第一个边界框，格式为 [x1, y1, w1, h1]
    :param box2: 第二个边界框，格式为 [x2, y2, w2, h2]
    :return: IoU 值
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算交集的左上角和右下角坐标
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    # 计算交集的面积
    inter_area = max(0, xB - xA) * max(0, yB - yA)

    # 计算两个边界框的面积
    box1_area = w1 * h1
    box2_area = w2 * h2

    # 计算并集的面积
    union_area = box1_area + box2_area - inter_area

    # 计算 IoU
    if union_area == 0:
        return 0
    return inter_area / union_area


def calculate_iou_for_frames(tracker_bboxes, gt_bboxes):
    """
    计算每一帧的 IoU
    :param tracker_bboxes: 跟踪器的边界框数组
    :param gt_bboxes: ground truth 的边界框数组
    :return: 每一帧的 IoU 值列表
    """
    iou_values = []
    for tracker_box, gt_box in zip(tracker_bboxes, gt_bboxes):
        iou = calculate_iou(tracker_box, gt_box)
        iou_values.append(iou)
    return iou_values


def plot_iou_comparison(trackers_iou, tracker_names,seq_name = None):
    """绘制5个跟踪器的IoU对比图（优化多曲线区分度）"""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 16,  # 增大字体大小
        "figure.dpi": 300,
        "figure.figsize": (12, 4),  # 增大图表宽度，减少高度至原来的三分之一多
        "axes.labelsize": 20,  # 增大坐标轴标签字体大小
        "axes.titlesize": 24,  # 增大标题字体大小
        "xtick.labelsize": 16,  # 增大 x 轴刻度标签字体大小
        "ytick.labelsize": 16,  # 增大 y 轴刻度标签字体大小
        "legend.fontsize": 16,  # 增大图例字体大小
        "axes.linewidth": 2  # 加粗坐标轴边框
    })

    frames = list(range(1, len(trackers_iou[0]) + 1))
    num_trackers = len(tracker_names)

    # ---------------------- 专业配色方案（5色，色盲友好） ---------------------- #
    # 使用ColorBrewer的"Set2"色系（8色，选前5色）
    # colors = ["#1f7c5f", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]  # 绿、橙、蓝、粉、黄绿
    colors = ["#FF0000", "#FF00FF", "#0000FF", "#FF7D7D", "#FFFF00"]  # 绿、橙、蓝、粉、黄绿
    # colors = ["#0000ff", "#ffff00", "#7d7dff", "#ff0000", "#ff00ff"]  # 绿、橙、蓝、粉、黄绿
    # 备用方案：seaborn默认调色板
    # colors = sns.color_palette("Set1", n_colors=5)

    # ---------------------- 线型组合（5种不同样式） ---------------------- #
    # linestyles = ["-", "--", ":", "-.", (0, (3, 5, 1, 5, 1, 5))]  # 实线、虚线、点线、点划线、长虚线
    linestyles = ["-", "-", "-", "-","-"]  # 实线、虚线、点线、点划线、长虚线

    fig, ax = plt.subplots()

    # Plot the red SACT curve last so it stays visible when curves overlap.
    draw_order = [i for i, name in enumerate(tracker_names) if name != "SACT"]
    draw_order.extend(i for i, name in enumerate(tracker_names) if name == "SACT")

    for i in draw_order:
        iou_values = trackers_iou[i]
        tracker_name = tracker_names[i]
        
        # ---------------------- 平滑处理（可选，突出趋势） ---------------------- #
        smoothed_iou = savgol_filter(iou_values, window_length=51, polyorder=3)  # 窗口长度建议为奇数（如51、101）
        
        # 绘制平滑曲线（主曲线，带标记）
        ax.plot(
            frames,
            smoothed_iou,
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=3,  # 加粗线条
            label=tracker_name
        )
        ax.plot(
            frames,
            iou_values,
            color=colors[i],
            alpha=0.3,
            linewidth=1  # 原始数据线条稍细
        )

    # ---------------------- 图表元素优化 ---------------------- #
    # ax.set(xlabel="Frame", ylabel="IoU", title="5 Trackers IoU Comparison (1000+ Frames)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # y轴刻度间隔0.1

    # ---------------------- 图例优化（下方居中布局） ---------------------- #
    # IoU curves are calculated against GroundTruth, so add it as a legend-only item.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#00FF00", linewidth=3, label="GroundTruth"))
    labels.append("GroundTruth")
    ax.legend(handles, labels, loc="upper center", ncol=len(labels),
              bbox_to_anchor=(0.5, -0.15), frameon=True, fancybox=True, shadow=True)
   
    # ax.legend(
    #     loc="upper center", ncol=len(tracker_names), bbox_to_anchor=(0.5, -0.05),  # 5列，置于图表下方
    #     frameon=True, fancybox=True, shadow=True, fontsize=12
    # )

    # ---------------------- x轴动态刻度（避免过密） ---------------------- #
    if len(frames) > 800:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20, integer=True))  # 最多显示20个刻度
    else:
        ax.xaxis.set_major_locator(ticker.AutoLocator())

    ax.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{seq_name[:-4]}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"图片已保存到: {output_path}")
    # plt.show()

def calculate_average(num_list):
    if not num_list:
        return 0
    total = sum(num_list)
    return total / len(num_list)




# ---------------------- 示例运行（5个跟踪器） ---------------------- #
# seq_name = '01_000000.txt'

seq_list = ['20190925_134301_1_9_IR.txt']

for seq_name in seq_list:

    gt_file_path = f"/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/groundtruth/{seq_name}"
    tracker1_file_path = f"/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/vitb_384_mae_ce_32x4_ep300_uav_wsp_best384/{seq_name}"
    tracker2_file_path = f"/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300_uav_lora/{seq_name}"
    tracker3_file_path = f"/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300_uav_adalora/{seq_name}"
    tracker4_file_path = f"/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300_uav_asc_lora_right/{seq_name}"
    tracker5_file_path = f"/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300_uav_milora/{seq_name}"
    tracker_files = [
        tracker1_file_path, tracker2_file_path, tracker3_file_path,
        tracker4_file_path, tracker5_file_path
    ]
    tracker_names = ["SACT", "FocusTrack", "MCITrack", "ODTrack", "DropTrack"]  # 跟踪器名称（与文件顺序一致）

    # 读取所有跟踪器数据
    gt_bboxes = read_bbox_file(gt_file_path)
    trackers_bboxes = [read_bbox_file(path) for path in tracker_files]

    # 计算所有跟踪器的IoU
    trackers_iou = [
        calculate_iou_for_frames(bboxes, gt_bboxes) for bboxes in trackers_bboxes
    ]
    mean_iou = [calculate_average(per_iou) for per_iou in trackers_iou]

    plot_iou_comparison(trackers_iou, tracker_names,seq_name = seq_name)
