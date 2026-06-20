"""
================================================================================
 OSTrack 逐层注意力可视化 + 输入扰动分析工具
================================================================================

核心目标：
  1. 可视化 12 个 Transformer block 的注意力分布（热力图）
  2. 通过输入扰动实验，判断每层依赖「局部纹理」还是「全局空间结构」
  3. 为微调提供「哪些层该改、哪些层该冻结」的数据支撑

原理说明：
  ── 注意力可视化 ──
  OSTrack 是单流架构：template (128×128) 和 search (256×256) 以 patch 形式
  concat 后一起送入 ViT backbone。每个 block 的 self-attention 矩阵中，
  search token 对 template token 的注意力强度，反映了该位置"有多像目标"。
  我们提取这部分注意力，reshape 成 16×16 热力图。

  ── 扰动分析 ──
  对 search 图像施加三种扰动，比较扰动前后的注意力变化量：
  · Patch Shuffle：打乱 8×8 小块 → 破坏局部纹理/颜色连续性，保留粗略形状
    → 如果某层对 patch shuffle 敏感，说明该层依赖局部 patch 关系（纹理层）
  · Gaussian Blur：高斯模糊 → 破坏高频纹理细节，保留低频轮廓和颜色
    → 如果某层对 blur 敏感，说明该层依赖高频纹理信息
  · Phase Scramble：FFT 相位随机化 → 保留颜色/纹理统计量，破坏空间结构
    → 如果某层对 phase scramble 敏感，说明该层依赖全局空间结构（语义层）

  核心指标：Δ = ‖A_pert − A_orig‖ / ‖A_orig‖（相对 Frobenius 变化量）
  Δ 越大 → 该层的注意力越依赖被破坏的那种特征

输出产物：
  ── 默认输出（4 张图）──
  1. attention_heatmaps.png          : 12 层纯热力图（3×4 grid）
  2. attention_overlay.png           : 热力图叠加在搜索图上
  3. attention_key_layers.png        : 5 个关键层特写（B0/B3/B6/B9/B11）
  4. attention_template.png          : 模板图参考

  ── --show_heads N 时额外输出 ──
  5. attention_layerN_heads.png      : 指定层 12 个 head 各自的注意力

  ── --perturb 时额外输出 ──
  6. attention_perturbation_sensitivity.png : 三层扰动 × 12 层敏感性曲线
  7. attention_perturbation_overview.png    : 原图+扰动图+柱状图综合
  8. attention_perturbation_data.json       : 完整数值（方便外部分析）

使用方法：
  # 基础用法（全图 + bbox，自动裁剪）
  python tools/vis_attention.py \
      --checkpoint <模型.pth> \
      --image <图片.jpg> \
      --bbox 298,247,44,34 \
      --output results/attention

  # 加扰动分析
  python tools/vis_attention.py ... --perturb

  # 加指定层 head 可视化
  python tools/vis_attention.py ... --show_heads 6

  # 更换模型配置
  python tools/vis_attention.py ... --config vitb_384_mae_ce_32x4_ep300_uav_oplora

作者：Claude Code
日期：2026-05-29
================================================================================
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，不弹窗
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 将项目根目录加入 sys.path，确保能 import lib 下的模块
prj_path = os.path.join(os.path.dirname(__file__), "..")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.models.ostrack import build_ostrack
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor


# ==============================================================================
#  工具函数
# ==============================================================================

def parse_bbox(s):
    """
    解析 bbox 字符串 'x,y,w,h' 或 'x y w h' 为浮点数列表。
    bbox 格式：[左上角 x, 左上角 y, 宽度, 高度]
    """
    parts = [float(x) for x in s.replace(",", " ").split()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 4 numbers: x y w h")
    return parts


def load_model(checkpoint_path: str, config_name: str = "vitb_256_mae_ce_32x4_ep300"):
    """
    加载训练好的 OSTrack 模型。

    步骤：
    1. 从 experiments/ostrack/<config_name>.yaml 读取模型配置
    2. 用 build_ostrack(cfg, training=False) 构建模型（不加载预训练权重）
    3. 加载指定 checkpoint 的 state_dict
    4. 移到 GPU，eval 模式
    """
    from lib.config.ostrack.config import cfg, update_config_from_file

    # 读取 YAML 配置到全局 cfg 对象（包含 backbone 类型、head 类型、CE 位置等）
    yaml_file = os.path.join(prj_path, "experiments", "ostrack", f"{config_name}.yaml")
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Config file not found: {yaml_file}")
    update_config_from_file(yaml_file)
    print(f"       Config: {yaml_file}")

    network = build_ostrack(cfg, training=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["net"], strict=True)
    network = network.cuda()
    network.eval()
    return network, cfg


def crop_with_sample_target(image, bbox, factor, output_sz):
    """
    用和 tracker 相同的逻辑从全图中裁剪目标区域。
    内部调用 sample_target，会做随机平移/缩放增强。

    参数：
        image    : [H, W, 3] numpy 全图
        bbox     : [x, y, w, h] 目标框
        factor   : 裁剪区域的放大倍数（2.0=模板, 4.0=搜索区域）
        output_sz: 输出正方形边长（128=模板, 256=搜索区域）

    返回：
        patch_arr  : [output_sz, output_sz, 3] 裁剪后的图
        resize_factor: 缩放因子
        amask_arr  : attention mask [output_sz, output_sz]
    """
    patch_arr, resize_factor, amask_arr = sample_target(
        image, bbox, factor, output_sz=output_sz
    )
    return patch_arr, resize_factor, amask_arr


# ==============================================================================
#  输入扰动函数 — 破坏图像的不同维度，探测每层的功能偏好
# ==============================================================================

def perturb_patch_shuffle(img, patch_size=8, seed=42):
    """
    Patch Shuffle（小块随机打乱）
    ────────────────────────────────
    将图像切成 patch_size×patch_size 的小块，然后随机打乱位置。

    破坏什么：局部纹理连续性、相邻 patch 之间的空间关系、颜色梯度
    保留什么：全局颜色分布、大尺度粗略形状

    如果某层注意力对这种扰动敏感 → 该层在做「局部 patch 到 patch 的关联」→ 纹理层
    """
    rng = np.random.RandomState(seed)
    H, W, C = img.shape

    # 将图像切成网格状的 patch
    patches = []
    positions = []
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patches.append(img[y:y + patch_size, x:x + patch_size].copy())
            positions.append((y, x))

    # 随机打乱 patch 顺序
    idx = rng.permutation(len(patches))
    patches = [patches[i] for i in idx]

    # 按打乱后的顺序贴回原位
    result = img.copy()
    for (y, x), patch in zip(positions, patches):
        result[y:y + patch_size, x:x + patch_size] = patch
    return result


def perturb_gaussian_blur(img, sigma=4.0):
    """
    Gaussian Blur（高斯模糊）
    ──────────────────────────
    用大 sigma 的高斯核对图像进行模糊。

    破坏什么：高频纹理细节（边缘、细纹路、噪点）
    保留什么：低频轮廓、颜色分布、大尺度结构

    如果某层对这种扰动敏感 → 该层依赖高频细节 → 高频纹理层
    """
    return cv2.GaussianBlur(img, (0, 0), sigma)


def perturb_phase_scramble(img, seed=42):
    """
    Phase Scramble（相位置乱）
    ───────────────────────────
    对每个通道做 FFT，保留幅度谱（magnitude），随机化相位谱（phase），
    再做 IFFT 重建图像。按通道独立处理是为了最大化对空间结构的破坏。

    保留什么：功率谱 = 纹理统计量（空间频率的分布），颜色分布
    破坏什么：所有空间结构信息（物体的位置、形状、朝向）

    注意：
    - 幅度谱保留意味着"图像里有多少边缘/纹理"不变
    - 相位随机意味着"边缘/纹理出现在哪里"完全混乱
    - 人眼看是同一堆纹理随机分布，但所有空间关系都消失了

    如果某层对这种扰动敏感 → 该层依赖物体的空间位置/形状关系 → 语义/结构层
    """
    rng = np.random.RandomState(seed)
    result = np.zeros_like(img, dtype=np.float64)
    for c in range(img.shape[2]):
        # 2D FFT 变换
        f = np.fft.fft2(img[:, :, c].astype(np.float64))
        # 提取幅度谱（保留）
        magnitude = np.abs(f)
        # 生成随机相位（破坏）
        phase = rng.uniform(-np.pi, np.pi, f.shape)
        # 合成 → IFFT → 取实部
        f_scrambled = magnitude * np.exp(1j * phase)
        ch = np.real(np.fft.ifft2(f_scrambled))
        # 归一化到 [0, 255] 范围
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8) * 255.0
        result[:, :, c] = ch
    return np.clip(result, 0, 255).astype(np.uint8)


# 扰动类型注册表：{名称: (函数, 图例标签)}
PERTURBATIONS = {
    "patch_shuffle":    (perturb_patch_shuffle,   "Patch Shuffle\n(texture disruption)"),
    "gaussian_blur":    (perturb_gaussian_blur,   "Gaussian Blur\n(high-freq disruption)"),
    "phase_scramble":   (perturb_phase_scramble,  "Phase Scramble\n(structure disruption)"),
}


# ==============================================================================
#  注意力采集 — 用 forward hook 从每个 block 抓取 attention 矩阵
# ==============================================================================

def collect_attention_maps(model, template_tensor, search_tensor):
    """
    用 PyTorch forward hook 从 12 个 CEBlock 中抓取注意力矩阵，
    提取 search→template 注意力并 reshape 为 2D 热力图。

    数据流：
      输入：template [1,3,128,128], search [1,3,256,256]
         ↓ patch embedding (16×16 conv, stride 16)
      template tokens [1, 64, 768]  +  search tokens [1, 256, 768]
         ↓ concat (direct mode)
      merged tokens [1, 320, 768]  (L_t=64, L_s=256)
         ↓ 12 × CEBlock (ce_keep_rate=1.0 禁用候选消除)
      self-attention: attn [1, 12, 320, 320]
         ↓ 取 search→template 部分: attn[:, :, 64:, :64] → [12, 256, 64]
         ↓ 对 head 和 template 维度取均值 → [256]
         ↓ reshape → [16, 16] 热力图

    为什么禁用 CE：
      CE 会在 B3/B6/B9 层消除 ∼30% 的 search token，导致不同层的 token 数量
      不一致（256→179→125→88），无法直接比较。设为 ce_keep_rate=1.0
      使所有层都保留完整的 256 个 token（16×16 grid）。

    参数：
        model            : OSTrack 模型实例
        template_tensor  : [1, 3, 128, 128] GPU tensor
        search_tensor    : [1, 3, 256, 256] GPU tensor

    返回：
        attn_maps       : list of [16, 16] np.ndarray × 12，每层一张热力图
        attn_heads_all  : list of [12, 16, 16] np.ndarray × 12，每层每个 head 的热力图
    """
    # 字典，用于存储 hook 捕获的注意力矩阵，key=层号
    attentions = {}

    def make_hook(layer_idx: int):
        """
        创建指定层的 forward hook 闭包。
        PyTorch 的 forward hook 在模块 forward() 执行后自动触发。
        CEBlock.forward() 返回 (x, global_index_t, global_index_s,
                                   removed_index_s, attn) 五元组，
        我们只取最后一个元素 attn。
        """
        def hook_fn(module, inputs, output):
            # CEBlock 返回 5 元组，attn 在最后
            if isinstance(output, (tuple, list)):
                attn = output[-1]  # shape: [B, num_heads, L_all, L_all]
            else:
                attn = None
            # 立即 .detach().cpu() 释放 GPU 显存
            attentions[layer_idx] = attn.detach().cpu() if attn is not None else None
        return hook_fn

    # 给 12 个 block 逐个注册 hook
    hooks = []
    for i, blk in enumerate(model.backbone.blocks):
        hook = blk.register_forward_hook(make_hook(i))
        hooks.append(hook)

    # 前向推理（ce_keep_rate=1.0 → 所有 block 保持 256 个 search token）
    with torch.no_grad():
        model(template=template_tensor, search=search_tensor, ce_keep_rate=1.0)

    # 推理结束后移除所有 hook（避免影响后续调用）
    for h in hooks:
        h.remove()

    # ── 处理每个 block 的注意力矩阵 ──
    num_blocks = len(model.backbone.blocks)
    attn_maps = []
    attn_heads_all = []
    L_t = 64  # template patch 数: (128/16)² = 8×8 = 64

    for i in range(num_blocks):
        attn = attentions.get(i)
        if attn is None:
            attn_maps.append(None)
            attn_heads_all.append(None)
            continue

        # attn shape: [B, num_heads, L_all, L_all]
        num_heads = attn.shape[1]
        L_s = attn.shape[-1] - L_t       # search patch 数，ce_keep_rate=1.0 时为 256
        grid_sz = int(L_s ** 0.5)         # grid 边长，应为 16

        # 安全检查：如果不是完整正方形网格则跳过（CE 造成的 token 数变化）
        if grid_sz * grid_sz != L_s:
            attn_maps.append(None)
            attn_heads_all.append(None)
            continue

        # ── 方法1: 所有 head 平均 → [16, 16] 综合热力图 ──
        # attn[:, :, L_t:, :L_t] 提取 search→template 部分: [B, H, L_s, L_t]
        attn_s_to_t = attn[0, :, L_t:, :L_t]  # [num_heads, L_s, L_t]
        # 对 head 维度 (dim=0) 和 template 维度 (dim=2) 取均值 → [L_s]=[256]
        attn_score = attn_s_to_t.mean(dim=(0, 2)).numpy()
        # reshape 为 [16, 16] 2D 热力图
        attn_maps.append(attn_score.reshape(grid_sz, grid_sz))

        # ── 方法2: 每个 head 独立 → [num_heads, 16, 16] ──
        # 只对 template 维度取均值（保留 head 维度）
        attn_s_to_t_per_head = attn[0, :, L_t:, :L_t]  # [H, L_s, L_t]
        attn_score_per_head = attn_s_to_t_per_head.mean(dim=2).numpy()  # [H, L_s]
        attn_heads_all.append(
            attn_score_per_head.reshape(num_heads, grid_sz, grid_sz)
        )

    return attn_maps, attn_heads_all


# ==============================================================================
#  扰动分析 — 核心实验逻辑
# ==============================================================================

def run_perturbation_analysis(model, preprocessor, template_img, template_amask,
                              search_img, search_amask, seed=42):
    """
    运行完整的扰动分析实验。

    流程：
    1. 预处理原始 search 图像 → 前向推理 → 记录 12 层原始注意力
    2. 对 search 图像依次施加 3 种扰动：
       a. Patch Shuffle（打乱 8×8 小块）
       b. Gaussian Blur（sigma=4.0 高斯模糊）
       c. Phase Scramble（FFT 相位随机化）
    3. 每种扰动后的图像单独前向推理 → 记录 12 层扰动后注意力
    4. 计算每层相对变化量：Δ = ‖A_pert − A_orig‖ / ‖A_orig‖

    注意：
    - 三种扰动只施加到 search 图像上，template 保持不变
    - 每次推理使用相同的 template（preprocess 一次，复用 tensor）
    - 为保证可重复性，随机种子从 seed 参数派生

    返回：
        diff_curves    : {"patch_shuffle": [12], "gaussian_blur": [12],
                          "phase_scramble": [12]}
                         每个值 = 该层在对应扰动下的注意力变化量
        perturbed_imgs : {"patch_shuffle": ndarray, ...}  扰动后的图像
    """
    # 预处理 template（只做一次，所有实验共享）
    template_tensor = preprocessor.process(template_img, template_amask).tensors

    # ── 原始图像：基准注意力 ──
    search_tensor = preprocessor.process(search_img, search_amask).tensors
    attn_orig, _ = collect_attention_maps(model, template_tensor, search_tensor)

    diff_curves = {}
    perturbed_imgs = {}

    # ── 依次运行三种扰动 ──
    for pert_name, (pert_fn, _) in PERTURBATIONS.items():
        # 为每种扰动生成独立的随机种子（从主种子派生）
        rng = np.random.RandomState(seed)
        pert_seed = rng.randint(0, 2**31)

        # 施加扰动
        if pert_name == "gaussian_blur":
            pert_img = pert_fn(search_img)              # Gaussian blur 不需要 seed
        else:
            pert_img = pert_fn(search_img, seed=pert_seed)

        perturbed_imgs[pert_name] = pert_img

        # 预处理扰动后的图像 → 前向推理 → 收集注意力
        pert_amask = np.ones(pert_img.shape[:2], dtype=np.float32)
        pert_tensor = preprocessor.process(pert_img, pert_amask).tensors
        attn_pert, _ = collect_attention_maps(model, template_tensor, pert_tensor)

        # 逐层计算注意力变化量：相对 Frobenius 范数
        diff_per_layer = []
        for i, (a_o, a_p) in enumerate(zip(attn_orig, attn_pert)):
            if a_o is None or a_p is None:
                diff_per_layer.append(np.nan)
            else:
                # Δ = ‖A_pert − A_orig‖_F / ‖A_orig‖_F
                change = np.linalg.norm(a_p - a_o) / (np.linalg.norm(a_o) + 1e-8)
                diff_per_layer.append(float(change))

        diff_curves[pert_name] = diff_per_layer

    return diff_curves, perturbed_imgs


# ==============================================================================
#  扰动结果可视化
# ==============================================================================

def plot_perturbation_sensitivity(diff_curves, search_img, perturbed_imgs,
                                  save_dir, tag="attention"):
    """
    将扰动分析结果生成两张图 + 控制台报告 + JSON 数据。

    图 A: attention_perturbation_sensitivity.png
          横轴=12个block，纵轴=注意力变化量Δ，三条曲线对应三种扰动
          红线(patch shuffle)→高在浅层 → 纹理层
          蓝线(phase scramble)→高在深层 → 语义层
          灰虚线标记 CE 层位置

    图 B: attention_perturbation_overview.png
          2行×4列：第一行=原图+3种扰动图，第二行=每层的敏感性柱状图
    """
    # 颜色和标记映射
    colors = {"patch_shuffle": "#e74c3c", "gaussian_blur": "#3498db", "phase_scramble": "#2ecc71"}
    markers = {"patch_shuffle": "o", "gaussian_blur": "s", "phase_scramble": "^"}

    n_layers = len(next(iter(diff_curves.values())))
    layers = np.arange(n_layers)

    # ═══════════════════════════════════════════════════════════════════
    # 图 A：核心敏感性曲线
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # 画三条曲线
    for pert_name, diff_vals in diff_curves.items():
        _, label = PERTURBATIONS[pert_name]
        ax.plot(layers, diff_vals,
                marker=markers[pert_name], color=colors[pert_name],
                label=label, linewidth=2, markersize=7)

    # 背景色标注"浅层区域"和"深层区域"（仅供参考，实际结果可能不同）
    ax.axvspan(-0.5, 2.5, alpha=0.06, color='#e74c3c')
    ax.axvspan(8.5, 11.5, alpha=0.06, color='#3498db')
    ax.text(1.0, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.95,
            "Shallow\n(texture-dominated)", ha='center', fontsize=8,
            color='#c0392b', style='italic')
    ax.text(10.0, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.95,
            "Deep\n(semantic-dominated)", ha='center', fontsize=8,
            color='#2980b9', style='italic')

    # CE 层标记（灰色虚线）
    for ce_layer in [3, 6, 9]:
        ax.axvline(x=ce_layer, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.text(ce_layer, -0.03, f'CE', ha='center', fontsize=7, color='gray',
                transform=ax.get_xaxis_transform())

    ax.set_xlabel("Block Layer", fontsize=12)
    ax.set_ylabel("Relative Attention Change  ‖A_pert − A_orig‖ / ‖A_orig‖", fontsize=11)
    ax.set_title("OSTrack — Layer-wise Attention Sensitivity to Input Perturbations\n"
                 "(higher = this layer's attention depends more on the disrupted feature)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(i) for i in layers])
    ax.grid(True, alpha=0.25, linestyle=':')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{tag}_perturbation_sensitivity.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════
    # 图 B：原图 + 三种扰动图 + 各自敏感性柱状图
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    all_imgs = [search_img] + [
        perturbed_imgs[p] for p in ["patch_shuffle", "gaussian_blur", "phase_scramble"]
    ]
    titles = ["Original", "Patch Shuffle", "Gaussian Blur", "Phase Scramble"]

    for col, (title, img) in enumerate(zip(titles, all_imgs)):
        # 第一行：图像本身
        axes[0, col].imshow(img)
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")
        axes[0, col].axis("off")

        # 第二行：该扰动类型的逐层敏感性柱状图
        pert_name = list(diff_curves.keys())[col - 1] if col > 0 else None
        if pert_name is not None:
            diff_vals = diff_curves[pert_name]
            # 按浅/中/深着色
            bar_colors = []
            for i in range(n_layers):
                if i <= 2:
                    bar_colors.append('#e74c3c')       # 浅层=红
                elif i >= 9:
                    bar_colors.append('#3498db')       # 深层=蓝
                else:
                    bar_colors.append('#95a5a6')       # 中层=灰
            axes[1, col].bar(layers, diff_vals, color=bar_colors, alpha=0.85)
            axes[1, col].set_xlabel("Layer", fontsize=9)
            axes[1, col].set_ylabel("Δ Attention", fontsize=9)
            axes[1, col].set_title(f"{title} — Sensitivity", fontsize=10)
            axes[1, col].set_xticks([0, 3, 6, 9, 11])
            axes[1, col].grid(True, alpha=0.3, linestyle=':')
        else:
            axes[1, col].axis("off")

    fig.suptitle("OSTrack — Input Perturbation Analysis\n"
                 "Which layers depend on texture vs. spatial structure?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{tag}_perturbation_overview.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Perturbation figures saved to: {save_dir}")

    # 打印控制台报告 + 保存 JSON
    _print_perturbation_summary(diff_curves)
    _save_perturbation_data(diff_curves, save_dir, tag)


# ==============================================================================
#  扰动报告 — 逐层数据表 + 总结 + 微调建议
# ==============================================================================

def _print_perturbation_summary(diff_curves):
    """
    在控制台打印详细分析报告，包含：
    1. 逐层数据表（12 行 × 4 列 + 判定）
    2. 浅/深层统计对比
    3. 纹理→语义转换点判定
    4. Top-5 最敏感层（综合 + 结构主导 + 纹理主导）
    """
    n_layers = len(next(iter(diff_curves.values())))

    # 提取三种扰动的 NumPy 数组（方便向量化计算）
    ps = np.array(diff_curves["phase_scramble"])      # 结构敏感性
    psh = np.array(diff_curves["patch_shuffle"])       # 纹理敏感性
    gb = np.array(diff_curves["gaussian_blur"])        # 高频敏感性

    # ── 逐层数据表 ──
    header = f"{'Layer':<7} {'PatchShuffle':>13} {'GaussBlur':>11} {'PhaseScram':>12} {'Dominant':>12}"
    sep = "─" * len(header)
    print("\n" + sep)
    print("  PER-LAYER PERTURBATION SENSITIVITY")
    print(sep)
    print(header)
    print("─" * len(header))

    for i in range(n_layers):
        dom = ""
        if not np.isnan(psh[i]) and not np.isnan(ps[i]):
            # 判定该层的主导特征类型
            diff = ps[i] - psh[i]
            if abs(diff) < 0.02:
                dom = "mixed"           # 差异小于 2%，标记为"混合"
            elif diff > 0:
                dom = "SEMANTIC ↑"      # 结构破坏影响 > 纹理破坏 → 结构/语义层
            else:
                dom = "texture  ↑"      # 纹理破坏影响 > 结构破坏 → 纹理层
        print(f"  Block {i:<2}  {psh[i]:>10.4f}     {gb[i]:>8.4f}    {ps[i]:>10.4f}     {dom:<12}")

    print("─" * len(header))

    # ── 浅层 vs 深层统计 ──
    shallow_tex = psh[:3].mean()         # Block 0-2 patch_shuffle 均值
    deep_tex = psh[9:].mean()            # Block 9-11 patch_shuffle 均值
    shallow_sem = ps[:3].mean()          # Block 0-2 phase_scramble 均值
    deep_sem = ps[9:].mean()             # Block 9-11 phase_scramble 均值

    # 第一个 phase_scramble >= patch_shuffle 的层 → 判断为纹理→语义转换点
    crossover = None
    for i in range(n_layers):
        if not np.isnan(ps[i]) and not np.isnan(psh[i]) and ps[i] >= psh[i] - 0.01:
            crossover = i
            break

    print(f"\n  Texture sensitivity  (patch shuffle)  shallow→deep:  {shallow_tex:.4f} → {deep_tex:.4f}")
    print(f"  Hi-freq sensitivity   (gaussian blur)   shallow→deep:  {gb[:3].mean():.4f} → {gb[9:].mean():.4f}")
    print(f"  Structure sensitivity (phase scramble)  shallow→deep:  {shallow_sem:.4f} → {deep_sem:.4f}")

    # 全层均值
    seg_texture = psh[~np.isnan(psh)]
    seg_semantic = ps[~np.isnan(ps)]
    print(f"\n  Mean patch_shuffle  across all layers:  {seg_texture.mean():.4f}")
    print(f"  Mean phase_scramble across all layers:  {seg_semantic.mean():.4f}")

    # ── 转换点判定 ──
    if crossover is not None:
        if crossover == 0:
            print(f"\n  ★ Semantic dominance from Block 0 — no texture→semantic transition.")
            print(f"    All blocks are structure-driven. Patch embedding eliminates texture signal.")
        else:
            print(f"\n  ★ Texture→Semantic transition at Block {crossover}")
            print(f"    Blocks 0–{crossover - 1}: texture-sensitive")
            print(f"    Blocks {crossover}–{n_layers - 1}: structure/semantic-sensitive")
    else:
        print(f"\n  ★ Texture dominance throughout — structure sensitivity never exceeds texture.")
        print(f"    Model relies primarily on local patch statistics, not global structure.")

    # ── 微调建议 ──
    print(f"\n  ── Micro-tuning recommendation ──")

    # 综合敏感性 Top-5（纹理+结构平均最高 → 最需要微调的层）
    combined = (psh + ps) / 2.0
    top_layers = np.argsort(combined)[::-1][:5]
    print(f"  Top-5 most perturbation-sensitive layers (highest Δ regardless of type):")
    for rank, layer_idx in enumerate(top_layers, 1):
        l = int(layer_idx)
        print(f"    #{rank} Block {l}:  texture={psh[l]:.4f}  structure={ps[l]:.4f}  combined={combined[l]:.4f}")

    # 结构主导 Top-5（phase_scramble >> patch_shuffle → 需要空间适配的层）
    ps_dominant = ps - psh
    top_semantic = np.argsort(ps_dominant)[::-1][:5]
    print(f"\n  Top-5 most structure-dominated layers (phase_scramble >> patch_shuffle):")
    for rank, layer_idx in enumerate(top_semantic, 1):
        l = int(layer_idx)
        print(f"    #{rank} Block {l}:  gap={ps_dominant[l]:.4f}  (Δsem={ps[l]:.4f}, Δtex={psh[l]:.4f})")

    # 纹理主导 Top-5（patch_shuffle >> phase_scramble → 需要外观适配的层）
    top_texture = np.argsort(-ps_dominant)[::-1][:5]
    print(f"\n  Top-5 most texture-dominated layers (patch_shuffle >> phase_scramble):")
    for rank, layer_idx in enumerate(top_texture, 1):
        l = int(layer_idx)
        print(f"    #{rank} Block {l}:  gap={-ps_dominant[l]:.4f}  (Δtex={psh[l]:.4f}, Δsem={ps[l]:.4f})")

    print("─" * len(header))


def _save_perturbation_data(diff_curves, save_dir, tag="attention"):
    """
    将扰动分析的完整数据导出为 JSON 文件。
    包含每层的原始数值 + 统计摘要 + 逐层判定，方便：
    - 外部画图（matplotlib/seaborn）
    - 做统计分析
    - 多模型对比
    """
    import json
    data = {
        "layers": list(range(len(next(iter(diff_curves.values()))))),
        "perturbations": {
            name: [float(v) if not np.isnan(v) else None for v in vals]
            for name, vals in diff_curves.items()
        },
        "summary": {}
    }

    ps = np.array(diff_curves["phase_scramble"])
    psh = np.array(diff_curves["patch_shuffle"])
    gb = np.array(diff_curves["gaussian_blur"])

    # 摘要统计
    data["summary"]["texture_shallow"] = float(psh[:3].mean())
    data["summary"]["texture_deep"] = float(psh[9:].mean())
    data["summary"]["structure_shallow"] = float(ps[:3].mean())
    data["summary"]["structure_deep"] = float(ps[9:].mean())
    data["summary"]["hi_freq_shallow"] = float(gb[:3].mean())
    data["summary"]["hi_freq_deep"] = float(gb[9:].mean())

    # 纹理→语义转换点
    crossover = None
    for i in range(len(ps)):
        if not np.isnan(ps[i]) and not np.isnan(psh[i]) and ps[i] >= psh[i] - 0.01:
            crossover = i
            break
    data["summary"]["crossover_layer"] = crossover

    # 逐层详细分析
    data["per_layer_analysis"] = []
    for i in range(len(ps)):
        if np.isnan(ps[i]) or np.isnan(psh[i]):
            data["per_layer_analysis"].append({"layer": i, "dominant": "N/A"})
        else:
            gap = float(ps[i] - psh[i])
            dom = "mixed" if abs(gap) < 0.02 else ("semantic" if gap > 0 else "texture")
            data["per_layer_analysis"].append({
                "layer": i,
                "patch_shuffle": float(psh[i]),
                "gaussian_blur": float(gb[i]),
                "phase_scramble": float(ps[i]),
                "gap": gap,
                "dominant": dom,
                "ce_layer": i in [3, 6, 9]
            })

    json_path = os.path.join(save_dir, f"{tag}_perturbation_data.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Raw data saved to: {json_path}")


# ==============================================================================
#  注意力可视化 — 4 张默认图
# ==============================================================================

def visualize_attention(attn_maps, search_img, template_img, save_dir, tag="attention"):
    """
    生成 4 张注意力可视化图。

    图 1: attention_heatmaps.png
          12 层纯热力图，3×4 grid。颜色越亮 = search 区域该位置对 template
          的注意力越强。所有层共享同一色阶（2%-98% 分位数），可直接比较。

    图 2: attention_overlay.png
          同图 1 的 12 层，但热力图以 55% 透明度叠加在搜索图像上，
          能直接看到"模型在看图像中的哪个区域"。

    图 3: attention_key_layers.png
          只展示 5 个关键层（B0/B3/B6/B9/B11），1×5 横排，
          CE 层有 [CE] 标记。这是最核心的一张。

    图 4: attention_template.png
          模板裁剪图（128×128），告诉读者模型在匹配什么目标。
    """
    os.makedirs(save_dir, exist_ok=True)
    n_layers = len(attn_maps)
    n_cols = 4                                               # 每行 4 列
    n_rows = (n_layers + n_cols - 1) // n_cols               # 3 行（12÷4）

    # ── 统一归一化：所有层共享色阶，基于 2%-98% 分位数裁剪 ──
    valid_maps = [m for m in attn_maps if m is not None]
    if not valid_maps:
        print("No valid attention maps to visualize!")
        return

    all_vals = np.concatenate([m.flatten() for m in valid_maps])
    vmin, vmax = np.percentile(all_vals, [2, 98])

    # ═══════════════════════════════════════════════════════════════════
    # 图 1：纯热力图（3×4 grid）
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i in range(n_layers):
        ax = axes[i]
        heatmap = attn_maps[i]
        if heatmap is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(f"Block {i}", fontsize=11)
            ax.axis("off")
            continue

        ax.imshow(heatmap, cmap="inferno", vmin=vmin, vmax=vmax,
                  interpolation="bilinear", aspect="equal")
        ax.set_title(f"Block {i}", fontsize=11, fontweight="bold")
        ax.axis("off")

    # 隐藏多余的子图位置
    for i in range(n_layers, len(axes)):
        axes[i].axis("off")

    # 右侧统一 colorbar
    fig.subplots_adjust(right=0.92, wspace=0.05, hspace=0.05)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(ScalarMappable(Normalize(vmin=vmin, vmax=vmax), "inferno"),
                        cax=cbar_ax)
    cbar.set_label("Mean Attention (search → template)", fontsize=10)

    fig.suptitle("OSTrack — Per-Layer Attention Maps\n"
                 "(search-region tokens attending to template tokens)",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(save_dir, f"{tag}_heatmaps.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════
    # 图 2：热力图叠加在原图上（3×4 grid）
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i in range(n_layers):
        ax = axes[i]
        heatmap = attn_maps[i]
        if heatmap is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(f"Block {i}", fontsize=11)
            ax.axis("off")
            continue

        # 将 16×16 热力图上采样到 256×256（搜索图分辨率）
        hmap_resized = cv2.resize(heatmap,
                                  (search_img.shape[1], search_img.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
        hmap_resized = np.clip((hmap_resized - vmin) / (vmax - vmin + 1e-8), 0, 1)

        # 叠加：α=0.55 热力图 + (1-α)=0.45 原图
        cmap = plt.cm.inferno
        hmap_colored = cmap(hmap_resized)[:, :, :3]  # RGBA → RGB（去掉 alpha 通道）
        alpha = 0.55
        overlay = ((1 - alpha) * (search_img.astype(np.float32) / 255.0)
                   + alpha * hmap_colored)
        overlay = np.clip(overlay, 0, 1)

        ax.imshow(overlay)
        ax.set_title(f"Block {i}", fontsize=11, fontweight="bold")
        ax.axis("off")

    for i in range(n_layers, len(axes)):
        axes[i].axis("off")

    fig.suptitle("OSTrack — Per-Layer Attention Overlay\n"
                 "(hotter = stronger attention to template)",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(save_dir, f"{tag}_overlay.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════
    # 图 3：5 个关键层特写（B0 / B3[CE] / B6[CE] / B9[CE] / B11）
    # ═══════════════════════════════════════════════════════════════════
    highlight_layers = [0, 3, 6, 9, 11]
    fig, axes = plt.subplots(1, len(highlight_layers),
                             figsize=(3.5 * len(highlight_layers), 3.5))

    for j, layer_idx in enumerate(highlight_layers):
        ax = axes[j]
        if layer_idx >= n_layers or attn_maps[layer_idx] is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center")
            ax.axis("off")
            continue

        heatmap = attn_maps[layer_idx]
        hmap_resized = cv2.resize(heatmap,
                                  (search_img.shape[1], search_img.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
        hmap_resized = np.clip((hmap_resized - vmin) / (vmax - vmin + 1e-8), 0, 1)
        hmap_colored = plt.cm.inferno(hmap_resized)[:, :, :3]

        alpha = 0.5
        overlay = ((1 - alpha) * (search_img.astype(np.float32) / 255.0)
                   + alpha * hmap_colored)
        overlay = np.clip(overlay, 0, 1)

        ax.imshow(overlay)
        ce_label = " [CE]" if layer_idx in [3, 6, 9] else ""
        ax.set_title(f"Block {layer_idx}{ce_label}", fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle("OSTrack — Attention Evolution at Key Layers",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(save_dir, f"{tag}_key_layers.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════
    # 图 4：模板图参考
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(template_img)
    ax.set_title("Template", fontsize=11, fontweight="bold")
    ax.axis("off")
    fig.savefig(os.path.join(save_dir, f"{tag}_template.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {4} figures to: {save_dir}")


# ==============================================================================
#  单层多头注意力可视化（--show_heads N）
# ==============================================================================

def visualize_per_head(attn_heads_all, layer_idx, search_img, save_dir, tag="attention"):
    """
    展示指定层的 12 个 attention head 各自关注什么。
    每个 head 独立归一化（热力图范围不同），因为不同 head 的注意力强度
    差异很大（有的强聚焦，有的均匀分布）。

    输出：3×4 grid，每个子图 = 一个 head 的注意力叠加在原图上
    """
    if layer_idx >= len(attn_heads_all) or attn_heads_all[layer_idx] is None:
        print(f"  No per-head data for layer {layer_idx}")
        return

    head_maps = attn_heads_all[layer_idx]  # [12, 16, 16] — 每个 head 一个热力图
    num_heads = head_maps.shape[0]
    cols = 4
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.2 * rows))
    axes = axes.flatten()

    search_bg = search_img.astype(np.float32) / 255.0

    for h in range(num_heads):
        ax = axes[h]
        heatmap = head_maps[h]

        # 上采样到搜索图尺寸
        hmap_resized = cv2.resize(heatmap,
                                  (search_img.shape[1], search_img.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
        # 每个 head 独立归一化（5%-95% 分位数）
        vmin_h, vmax_h = np.percentile(hmap_resized, [5, 95])
        hmap_norm = np.clip((hmap_resized - vmin_h) / (vmax_h - vmin_h + 1e-8), 0, 1)

        # 叠加：α=0.55 热力图 + 0.45 原图
        hmap_colored = plt.cm.inferno(hmap_norm)[:, :, :3]
        overlay = 0.45 * search_bg + 0.55 * hmap_colored
        overlay = np.clip(overlay, 0, 1)
        ax.imshow(overlay)
        ax.set_title(f"Head {h}", fontsize=9)
        ax.axis("off")

    for h in range(num_heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle(f"OSTrack — Block {layer_idx}: Per-Head Attention (search → template)",
                 fontsize=12, fontweight="bold")
    out_path = os.path.join(save_dir, f"{tag}_layer{layer_idx}_heads.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Per-head figure saved to: {out_path}")


# ==============================================================================
#  主函数 — 命令行入口
# ==============================================================================

def main():
    """
    OSTrack 注意力可视化 & 扰动分析工具的命令行入口。

    两阶段流程：
      Phase 1（必有）：加载模型 + 准备图像 + 前向推理 → 4 张注意力热力图
      Phase 2（可选）：--perturb → 扰动分析（+3 次前向推理）
      Phase 3（可选）：--show_heads N → 指定 layer 的 12-head 图
    """
    parser = argparse.ArgumentParser(
        description="OSTrack per-layer attention visualization"
    )

    # ── 模型相关 ──
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径 (.pth.tar)")
    parser.add_argument("--config", type=str, default="vitb_256_mae_ce_32x4_ep300",
                        help="YAML 配置名（不含路径和扩展名），默认 vitb_256_mae_ce_32x4_ep300")

    # ── 图像输入（两种方式二选一）──
    parser.add_argument("--image", type=str, default=None,
                        help="完整帧图像路径（配合 --bbox 使用）")
    parser.add_argument("--bbox", type=parse_bbox, default=None,
                        help="目标边界框 x y w h（配合 --image 使用）")
    parser.add_argument("--template", type=str, default=None,
                        help="预裁剪的模板图像路径（128×128）")
    parser.add_argument("--search", type=str, default=None,
                        help="预裁剪的搜索图像路径（256×256）")

    # ── 输出相关 ──
    parser.add_argument("--output", type=str, default="results/attention",
                        help="输出目录，默认 results/attention")
    parser.add_argument("--search_area_factor", type=float, default=4.0,
                        help="搜索区域相对于目标的放大倍数，默认 4.0")

    # ── 可选功能 ──
    parser.add_argument("--show_heads", type=int, default=None,
                        help="展示指定层 (0-11) 的 12 个 head 各自注意力")
    parser.add_argument("--perturb", action="store_true", default=False,
                        help="运行输入扰动分析（patch shuffle / blur / phase scramble）")

    args = parser.parse_args()

    # ── 验证输入模式 ──
    if args.template and args.search:
        mode = "crops"
    elif args.image and args.bbox:
        mode = "image"
    else:
        print("ERROR: Provide either (--template + --search) or (--image + --bbox)")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # 阶段 1：加载模型
    # ═══════════════════════════════════════════════════════════════════
    print(f"[1/4] Loading model from: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config)
    print(f"       Backbone: {cfg.MODEL.BACKBONE.TYPE}, "
          f"blocks: {len(model.backbone.blocks)}")

    # 图像预处理器（负责归一化、转 tensor、上 GPU）
    preprocessor = Preprocessor()

    # ═══════════════════════════════════════════════════════════════════
    # 阶段 2：准备图像
    # ═══════════════════════════════════════════════════════════════════
    if mode == "crops":
        print(f"[2/4] Loading pre-cropped images")
        template_img = cv2.imread(args.template)
        search_img = cv2.imread(args.search)
        if template_img is None:
            print(f"ERROR: Cannot read template image: {args.template}")
            sys.exit(1)
        if search_img is None:
            print(f"ERROR: Cannot read search image: {args.search}")
            sys.exit(1)
        # BGR → RGB
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)

        # 全 1 mask（不做注意力遮蔽）
        amask_t = np.ones(template_img.shape[:2], dtype=np.float32)
        amask_s = np.ones(search_img.shape[:2], dtype=np.float32)

    else:  # mode == "image"
        print(f"[2/4] Loading image and cropping")
        full_img = cv2.imread(args.image)
        if full_img is None:
            print(f"ERROR: Cannot read image: {args.image}")
            sys.exit(1)
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
        H, W = full_img.shape[:2]

        bbox_xywh = args.bbox  # [x, y, w, h]
        print(f"       Image: {W}x{H}, bbox: {bbox_xywh}")

        # 裁剪模板区域（2× 放大，128×128 输出）
        template_factor = 2.0
        template_sz = 128
        t_patch, t_rf, t_amask = crop_with_sample_target(
            full_img, bbox_xywh, template_factor, template_sz
        )
        template_img = t_patch
        amask_t = t_amask

        # 裁剪搜索区域（4× 放大，256×256 输出）
        search_factor = args.search_area_factor
        search_sz = 256
        s_patch, s_rf, s_amask = crop_with_sample_target(
            full_img, bbox_xywh, search_factor, search_sz
        )
        search_img = s_patch
        amask_s = s_amask

    print(f"       Template: {template_img.shape}, Search: {search_img.shape}")

    # 图像预处理：归一化 + 转 GPU tensor
    template_data = preprocessor.process(template_img, amask_t)
    search_data = preprocessor.process(search_img, amask_s)

    # ═══════════════════════════════════════════════════════════════════
    # 阶段 3：前向推理 + 收集注意力
    # ═══════════════════════════════════════════════════════════════════
    print(f"[3/4] Running forward pass to collect attention maps...")
    attn_maps, attn_heads_all = collect_attention_maps(
        model,
        template_data.tensors,
        search_data.tensors,
    )

    valid_count = sum(1 for m in attn_maps if m is not None)
    print(f"       Captured {valid_count}/{len(attn_maps)} attention maps")

    # ═══════════════════════════════════════════════════════════════════
    # 阶段 4：生成可视化
    # ═══════════════════════════════════════════════════════════════════
    print(f"[4/4] Generating visualizations...")
    visualize_attention(attn_maps, search_img, template_img, args.output)

    # ── 可选：单层 12-head 可视化 ──
    if args.show_heads is not None:
        visualize_per_head(attn_heads_all, args.show_heads, search_img, args.output)

    # ── 可选：扰动分析 ──
    if args.perturb:
        print(f"\n[EXTRA] Running perturbation analysis...")
        print(f"        This runs 3 extra forward passes (patch shuffle / blur / phase scramble)")
        diff_curves, perturbed_imgs = run_perturbation_analysis(
            model, preprocessor, template_img, amask_t,
            search_img, amask_s,
        )
        plot_perturbation_sensitivity(diff_curves, search_img, perturbed_imgs,
                                      args.output)

    print("Done!")


if __name__ == "__main__":
    main()
