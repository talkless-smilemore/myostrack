"""
================================================================================
Weighted Spectral Projection (WSP) — 反无人机小目标适配器  (Anti-UAV Adapter)
================================================================================

本文件是 UAV-WSP 的核心实现，包含以下部分（按代码顺序）：

    1. compute_spectral_weights()       —— 计算谱衰减系数 d_i = (σ_i/σ₁)^β
    2. _detail_saliency_from_svd()      —— 计算门控初始化偏置（细节显著性）
    3. WSP_DEFAULT_PRIOR_CONFIG         —— 默认层配置（基于扰动实验证据）
    4. WeightedSpectralProjection 类     —— WSP 适配器模块（核心）
       ├── set_progress()               —— 设置全局训练进度（类方法）
       ├── __init__()                   —— 初始化：SVD + 投影 + 门控 + 低秩矩阵
       ├── from_linear()                —— 工厂方法：从 nn.Linear 创建 WSP 模块
       ├── forward()                    —— 前向传播（加权投影 + 门控）
       ├── _gini_coefficient()          —— Gini 系数计算（静态方法）
       ├── _schedule_lam()              —— 正则化权重调度
       ├── regularisation_loss()        —— 计算正则化损失
       └── merge_to_weight()            —— 合并到权重（推理用）
    5. collect_wsp_regularisation()     —— 遍历模型收集所有 WSP 模块的正则损失
    6. inject_wsp_into_backbone()       —— 注入 WSP 适配器到 ViT backbone
    7. _get_linear()                    —— 工具函数：按名字查找 Linear 层
    8. _count_frozen()                  —— 工具函数：统计冻结的 block 数

设计起源（反无人机动机）：
    先前的 PEFT 方法（OPLoRA、SGLoRA）使用硬正交投影 P = I - UUᵀ，
    完全移除 top-k 奇异方向的适应能力。但小目标的信息（边缘、纹理、
    高频边界响应）恰恰由中排名的奇异方向承载——硬投影会堵塞这些方向。

    WSP 用加权投影 P = I - UDUᵀ 替换硬投影，其中 D 是渐变衰减的对角矩阵，
    d_i = (σ_i/σ₁)^β。σ₁ 方向几乎完全保护（d₁≈1），而 σ_k 方向保留大部分
    适应自由（d_k≈(σ_k/σ₁)^β），实现了细节敏感度的连续渐变保护。

核心公式：
    ΔW = (α/r) · diag(g) · P_L(w) · B · A · P_R(w)
    P_R(w) = I - V_k D V_k^T    （加权输入投影）
    P_L(w) = I - U_k D U_k^T    （加权输出投影）
    D = diag(d_1, ..., d_k)     （谱权重矩阵）
    d_i = (σ_i / σ_1)^β         （β ≥ 0，默认 1.0）

正则化：
    L_reg = λ_e·H(σ(s)) + λ_g·Σ||B_i||₂ - λ_f·G(σ(s))
           熵（二值化门控）  group-lasso（神经元剪枝）  Gini 聚焦（集中）
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral weight helpers
# ---------------------------------------------------------------------------

# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数 1：计算谱权重（SpectrAl Weight）
# ──────────────────────────────────────────────────────────────────────────────
# 这是加权谱投影的核心：d_i = (σ_i / σ₁)^β
#   β=0 → 退化为硬投影（所有 d_i=1）
#   β=1 → 线性衰减（默认）
#   β<1 → 更慢衰减 = 更多适应自由（小目标友好）
#   β>1 → 更快衰减 = 更保守
# ──────────────────────────────────────────────────────────────────────────────

def compute_spectral_weights(
    S: torch.Tensor,
    k: int,
    beta: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    计算谱衰减系数 d_i = (σ_i/σ₁)^β，用于加权投影。

    参数:
        S:    W₀ 的奇异值  [min(d_out, d_in)]，从大到小排列
        k:    要保护的 top 奇异方向数量
        beta: 衰减指数。
              β=0 → 全部 d_i=1 → 退化为原始硬投影 P = I-UUᵀ
              β=1 → d_i = σ_i/σ₁ → 线性衰减（默认）
              β>1 → 更快衰减 → 对大奇异值方向保护更强
              β<1 → 更慢衰减 → 小目标细节有更多适应自由
        eps:  数值稳定性

    返回:
        d:  谱权重  [k]，范围 (0, 1]，d₁=1，dₖ=(σₖ/σ₁)^β
    """
    sigma_1 = S[0]                     # 最大奇异值
    # 关键是除法 σ_i/σ₁：大奇异值 ≈ 1 → 权重 ≈ 1（受保护）
    #              小奇异值 << 1 → 权重 << 1（保留适应自由）
    d = (S[:k] / (sigma_1 + eps)).pow(beta).clamp(0.0, 1.0)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数 2：细节显著性门控初始化（Detail-Saliency Gate Initialization）
# ──────────────────────────────────────────────────────────────────────────────
# 动机：与小奇异值（细节/边缘/纹理）方向耦合强的输出通道，应该初始门控偏高，
#       让模型在训练一开始就偏向使用这些通道来处理小目标。
#
# 计算公式：
#   detail_i = Σ_j (Uk[i,j]² · (1 - d_j))
#     其中 Uk[i,j]² = 输出通道 i 在奇异方向 j 上的参与度
#          (1-d_j)  = 奇异方向 j 的"细节含量"（小奇异值 → 1-d_j 大 → 细节多）
#   gate_logit_init = 0.6 · normalize(detail) - 0.3  → 范围 [-0.3, 0.3]
#     正偏置 → sigmoid ≈ 0.55~0.65（初始打开，小目标通道）
#     负偏置 → sigmoid ≈ 0.35~0.45（初始关闭，背景通道）
# ──────────────────────────────────────────────────────────────────────────────

def _detail_saliency_from_svd(
    Uk: torch.Tensor,
    S: torch.Tensor,
    k: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    计算每个输出通道的"细节显著性"偏置，用于门控初始化。

    思路：输出通道如果与小的奇异值方向（细节/纹理）耦合强，
          就给它的门控一个正的初始偏置，让小目标特征有"头马效应"。

    参数:
        Uk:   左奇异矩阵  [d_out, k]，每列是一个奇异方向
        S:    奇异值  [k]
        k:    top-k 数量
        beta: 谱衰减指数

    返回:
        bias: 每个输出通道的偏置  [d_out]，范围 [-0.3, 0.3]
    """
    d = compute_spectral_weights(S, k, beta)            # [k]  谱权重
    # 关键计算：detail_i = Σ Uk[i,j]² · (1 - d_j)
    # Uk[i,j]² 大 + (1-d_j) 大 → 通道 i 与细节方向耦合强 → 正偏置
    detail = (Uk.pow(2) * (1.0 - d).unsqueeze(0)).sum(dim=1)  # [d_out]
    # 归一化到 [-0.3, 0.3] — 小偏置，不压倒任务损失
    d_max = detail.max()
    if d_max > 1e-10:
        detail = 0.6 * detail / d_max - 0.3
    else:
        detail = torch.zeros_like(detail)
    return detail


# ──────────────────────────────────────────────────────────────────────────────
# 默认层配置（基于扰动实验证据）
# ──────────────────────────────────────────────────────────────────────────────
# 通过 PatchShuffle + GaussBlur + PhaseScramble 三重扰动实验测量 ViT 各层
# 对反无人机追踪的敏感性，得分 = 综合敏感性（越高越重要）。
#
#   B6: 1.559 (最高) — CE 综合决策层，GaussBlur 峰值
#   B7: 1.491         — 融合枢纽，纹理+结构双轨
#   B9: 1.479         — CE 纹理筛选，PatchShuffle 峰值
#   B5: 1.380         — CE 后局部重建
#   B1: 1.224         — 纹理点火（4.6× 跳变）
#   B3: 1.170         — CE 空间筛选入口
#   B2: 1.061         — 全局空间构建
#   ★ B0(embedding), B11(稳定输出) → 冻结
#
# 分组策略（V2）：
#   A组: 核心适配 — B7(融合)+B1(纹理), rank=12, β=0.5, 全target
#   B组: CE决策 — B6+B9+B3, rank=12, β=0.5, 仅attention
#   C组: 结构支持 — B5+B2, rank=8, β=0.5, 仅MLP
#   D组: 遮挡处理 — B8, rank=2, β=0.3, 几乎不保护
#   E组(V2新增): 轻量解冻 — B4+B10, rank=2, β=0.3
#   冻结: B0, B11
# ──────────────────────────────────────────────────────────────────────────────

WSP_DEFAULT_PRIOR_CONFIG: List[Dict[str, Any]] = [
    # Group A: core adaptation — rank=12, β=0.5 (relaxed protection)
    #   B7: fusion hub (comprehensive=1.491), texture+structure dual-track
    #   B1: texture ignition (comprehensive=1.224), 4.6× PatchShuffle jump
    {
        "blocks": [7, 1],
        "rank": 12,
        "top_k": 16,
        "alpha": 8.0,
        "spectral_beta": 0.5,
        "targets": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
    },
    # Group B: CE decision layers — rank=12, attention-focused, β=0.5
    #   B6: comprehensive decision (comprehensive=1.559, GaussBlur peak 0.538)
    #   B9: texture screening (comprehensive=1.479, PatchShuffle peak 0.713)
    #   B3: spatial screening (comprehensive=1.170, CE entry)
    {
        "blocks": [6, 9, 3],
        "rank": 12,
        "top_k": 16,
        "alpha": 8.0,
        "spectral_beta": 0.5,
        "targets": ["attn.qkv", "attn.proj"],
    },
    # Group C: structural support — rank=8, MLP-only, β=0.5
    #   B5: post-CE reconstruction (comprehensive=1.380, two-metric dual-high)
    #   B2: global spatial construction (comprehensive=1.061, new-view adaptation)
    {
        "blocks": [5, 2],
        "rank": 8,
        "top_k": 16,
        "alpha": 8.0,
        "spectral_beta": 0.5,
        "targets": ["mlp.fc1", "mlp.fc2"],
    },
    # Group D: occlusion/edge-case support — rank=2, β=0.3
    #   B8: pure local texture (comprehensive=0.843), nearly unprotected for occlusion
    {
        "blocks": [8],
        "rank": 2,
        "top_k": 16,
        "alpha": 8.0,
        "spectral_beta": 0.3,
        "targets": ["attn.qkv", "attn.proj"],
    },
    # Group E (NEW): light thaw — rank=2, β=0.3, attention+MLP-fc1
    #   B4: spatial refinement (comprehensive=1.089), formerly frozen
    #   B10: fine-tuning (comprehensive=1.246), formerly frozen
    {
        "blocks": [4, 10],
        "rank": 2,
        "top_k": 16,
        "alpha": 8.0,
        "spectral_beta": 0.3,
        "targets": ["attn.qkv", "mlp.fc1"],
    },
    # Frozen: B0 (embedding), B11 (stable output)
]


# ──────────────────────────────────────────────────────────────────────────────
# 核心类：WeightedSpectralProjection（加权谱投影适配器模块）
# ──────────────────────────────────────────────────────────────────────────────
# 这是 WSP 的构建块，替换 ViT backbone 中的一个 nn.Linear 层。
#
# 整体架构：
#   冻结路径（frozen path）：out = x @ W₀ᵀ + b       ← 原始权重不更新
#   适配器路径（adapter path）：
#     x_r = P_R(w) · x                              ← 加权输入投影
#     z   = x_r @ Aᵇ                                ← 低秩细节编码 (r维瓶颈)
#     u   = z @ Bᵇ                                  ← 目标特异性解码 (d_out维)
#     u_r = P_L(w) · u                              ← 加权输出投影
#     u_g = σ(s) · u_r                              ← 目标显著性门控（可学习）
#     Δy  = (α/r) · u_g                             ← 缩放残差
#   总输出 = frozen + adapter
#
# 关键创新：
#   1. 加权谱投影 P = I - UDUᵀ（不是硬切断）
#   2. 细节显著性门控初始化（小目标通道初始打开）
#   3. 三项正则化：熵二值化 + Group-Lasso 剪枝 + Gini 聚焦
# ──────────────────────────────────────────────────────────────────────────────

class WeightedSpectralProjection(nn.Module):
    """
    单个 WSP 适配器模块——反无人机 PEFT 的构建块。

    替换 ViT backbone 中的一个 nn.Linear 层。冻结原始权重 W₀，
    通过低秩适配器 + 加权谱投影 + 可学习门控提供参数高效的更新。

    关键参数:
        spectral_beta: 谱衰减指数。控制投影对 top-k 奇异方向的保护渐变程度。
            β=1 是好的默认值；低值 (0.3-0.7) 给更多适应自由；高值 (1.5-2.0) 更保守。
        focus_lam_max: Gini 聚焦正则化峰值强度。推动门控稀疏集中。
            对需要高度选择性特征的小目标有用。

    前向（行向量约定）:
        out  = x @ W0^T + bias                       … 冻结基路径
        x_r  = x - (x @ V_k) · D · V_k^T             … 加权输入投影
        z    = x_r @ A^T                               … 低秩细节编码
        u    = z @ B^T                                 … 目标特异性解码
        u_r  = u - (u @ U_k) · D · U_k^T              … 加权输出投影
        u_g  = sigmoid(s) * u_r                        … 目标显著性门控
        out += (alpha / rank) * u_g                    … 任务适应残差
    """

    # ══════════════════════════════════════════════════════════════════════════
    # 类级训练进度追踪（全局共享）
    # ══════════════════════════════════════════════════════════════════════════
    # _global_progress 是所有 WSP 实例共享的类变量，范围 [0, 1]。
    # 每个 epoch 由 OSTrackActor 调用 set_progress() 设置。
    # 用在 _schedule_lam() 中控制正则化权重从 0 到最大值的调度。
    # ══════════════════════════════════════════════════════════════════════════
    _global_progress: float = 0.0  # [0, 1]  当前 epoch / 总 epoch

    @classmethod
    def set_progress(cls, progress: float) -> None:
        """
        设置训练进度 [0, 1]，所有 WSP 实例共享。

        由 OSTrackActor 在每个 epoch 的前向传播前调用：
            progress = epoch / total_epochs
            WeightedSpectralProjection.set_progress(progress)

        这会影响所有 WSP 模块的 _schedule_lam()，从而控制正则化权重。
        """
        cls._global_progress = max(0.0, min(1.0, float(progress)))

    # ══════════════════════════════════════════════════════════════════════════
    # 初始化（__init__）
    # ══════════════════════════════════════════════════════════════════════════
    # 按顺序做了 8 件事：
    #   1. 保存超参数（rank, alpha, top_k, spectral_beta, 正则化参数）
    #   2. 注册冻结的 weight 和 bias（frozen path用，不更新）
    #   3. 若 rank<=0 → 创建空壳，不启动适配器
    #   4. SVD 分解 W₀ → 取 top-k 奇异方向和对应的 U_k, V_k
    #   5. 计算谱权重 D = diag(d_i), d_i = (σ_i/σ₁)^β
    #   6. 计算细节显著性门控偏置（小目标通道初始略打开）
    #   7. 初始化低秩矩阵 A（kaiming 均匀）和 B（全零）
    #   8. 初始化门控 logits = 细节偏置
    # ══════════════════════════════════════════════════════════════════════════

    def __init__(
        self,
        in_features: int,      # 输入维度
        out_features: int,     # 输出维度
        bias: bool,            # 原始 Linear 是否有 bias
        rank: int,             # 低秩瓶颈维度 r（越小越高效）
        top_k: int,            # 要保护的 top 奇异方向数
        alpha: float,          # 适配器缩放因子
        weight: torch.Tensor,  # 原始 Linear 的权重 W₀
        bias_tensor: Optional[torch.Tensor],  # 原始 Linear 的 bias
        # 反无人机创新：谱衰减指数
        spectral_beta: float = 1.0,
        # 熵正则化调度
        entropy_lam_max: float = 1e-4,
        warmup_ratio: float = 0.33,
        anneal_ratio: float = 0.33,
        # B 矩阵行 Group-Lasso
        group_lasso_lam_max: float = 1e-5,
        # 聚焦正则化（反无人机：稀疏目标显著性集中）
        focus_lam_max: float = 1e-5,
    ):
        super().__init__()

        # ── 第 1 步：保存所有超参数 ────────────────────────────────────────
        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.top_k = int(top_k)
        self.spectral_beta = float(spectral_beta)

        # 正则化调度参数
        self.entropy_lam_max = float(entropy_lam_max)
        self.warmup_ratio = float(warmup_ratio)
        self.anneal_ratio = float(anneal_ratio)
        self.group_lasso_lam_max = float(group_lasso_lam_max)
        self.focus_lam_max = float(focus_lam_max)

        # ── 第 2 步：注册冻结的 weight 和 bias ─────────────────────────────
        # detach() + requires_grad=False → 这两项不参与梯度更新
        # 这是 frozen path 用的：out = x @ W₀ᵀ + b
        self.register_parameter(
            "weight", nn.Parameter(weight.detach(), requires_grad=False)
        )
        if bias and bias_tensor is not None:
            self.register_parameter(
                "bias", nn.Parameter(bias_tensor.detach(), requires_grad=False)
            )
        else:
            self.register_parameter("bias", None)

        # ── 第 3 步：rank<=0 保护 ──────────────────────────────────────────
        # 如果配置要求 rank=0，创建一个空壳模块（forward 直接返回 out）
        # 所有 buffer 和 parameter 都创建空张量，确保 optimizer 能处理
        if rank <= 0:
            self._active = False
            self.register_buffer(
                "spectral_weight", torch.empty(0, device=weight.device, dtype=weight.dtype)
            )
            self.register_buffer(
                "Uk", torch.empty(out_features, 0, device=weight.device, dtype=weight.dtype)
            )
            self.register_buffer(
                "Vk", torch.empty(in_features, 0, device=weight.device, dtype=weight.dtype)
            )
            self.register_parameter(
                "lora_A", nn.Parameter(torch.zeros(0, in_features))
            )
            self.register_parameter(
                "lora_B", nn.Parameter(torch.zeros(out_features, 0))
            )
            self.register_parameter(
                "gate_logit", nn.Parameter(torch.zeros(out_features))
            )
            return

        self._active = True

        # ── 第 4 步：SVD 分解，取 top-k 奇异方向 ─────────────────────────────
        # 对冻结权重 W₀ 做奇异值分解：W₀ = U Σ Vᵀ
        # 取 top-k 个奇异方向用于构建加权投影矩阵
        #   U_k [d_out, k] — 左奇异向量（输出通道的"模式"）
        #   V_k [d_in, k]  — 右奇异向量（输入通道的"模式"）
        #   σ [k]          — 奇异值（从大到小排列）
        W = self.weight.data
        kk = min(top_k, min(out_features, in_features))
        if kk > 0:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            kk = min(kk, U.shape[1], Vh.shape[0])
            Uk = U[:, :kk]             # [d_out, k]
            Vk = Vh[:kk, :].T           # [d_in, k]
            sigma = S[:kk]              # [k]

            # ── 第 5 步：计算谱权重 D = diag(d_i) ──────────────────────────
            # d_i = (σ_i / σ₁)^β  ∈ [0, 1]
            # 这是 WSP 的核心反无人机创新：渐变保护代替硬切断。
            #   σ₁ → d₁=1（完全保护，不让 LoRA 改变背景模式）
            #   σₖ → dₖ=(σₖ/σ₁)^β（小值 → 保留适应空间供小目标细节使用）
            d = compute_spectral_weights(sigma, kk, self.spectral_beta)

            # ── 第 6 步：计算细节显著性门控初始化偏置 ───────────────────────
            # 输出通道如果与小奇异值（细节）方向耦合强 → 初始门控偏高
            # 让模型一开始就倾向于用这些通道处理小目标
            gate_bias = _detail_saliency_from_svd(Uk, sigma, kk, self.spectral_beta)
            # 初始化 logit = 偏置（sigmoid(0)=0.5 是中性值，这里用偏置偏离中性）
            gate_logit_init = gate_bias
        else:
            # top_k=0 → 无投影
            Uk = torch.empty(out_features, 0, device=W.device, dtype=W.dtype)
            Vk = torch.empty(in_features, 0, device=W.device, dtype=W.dtype)
            d = torch.empty(0, device=W.device, dtype=W.dtype)
            gate_logit_init = torch.zeros(out_features, device=W.device, dtype=W.dtype)

        # 注册为 buffer（不会更新，但会随模型保存和加载）
        self.register_buffer("spectral_weight", d.contiguous())   # [k]
        self.register_buffer("Uk", Uk.contiguous())                # [d_out, k]
        self.register_buffer("Vk", Vk.contiguous())                # [d_in, k]

        # ── 第 7 步：初始化低秩适配器矩阵 A 和 B ───────────────────────────
        #   lora_A [rank, d_in]   — 低秩编码（输入→瓶颈）
        #   lora_B [d_out, rank]  — 目标解码（瓶颈→输出）
        #   A 用 kaiming 均匀初始化（保证梯度良好）
        #   B 用零初始化（一开始适配器不改变输出，与 pretrained 权重一致）
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # ── 第 8 步：初始化门控 logits ──────────────────────────────────────
        # 每个输出通道一个门控值，sigmoid 映射到 (0, 1)
        # 初始化加了细节偏置：小目标通道初始 ~0.55-0.65，背景通道 ~0.35-0.45
        self.gate_logit = nn.Parameter(gate_logit_init)

    # ══════════════════════════════════════════════════════════════════════════
    # 工厂方法：从 nn.Linear 创建 WSP 模块
    # ══════════════════════════════════════════════════════════════════════════
    # inject_wsp_into_backbone() 在遍历 ViT 的层时，遇到 nn.Linear 就调用
    # 这个方法把它包装成 WeightedSpectralProjection 模块。
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        top_k: int,
        alpha: float,
        spectral_beta: float = 1.0,
        entropy_lam_max: float = 1e-4,
        warmup_ratio: float = 0.33,
        anneal_ratio: float = 0.33,
        group_lasso_lam_max: float = 1e-5,
        focus_lam_max: float = 1e-5,
    ) -> "WeightedSpectralProjection":
        """
        从一个已有的 nn.Linear 层创建 WSP 适配器模块。

        做的事情很简单：把 Linear 的 weight 和 bias 提取出来，
        传入 __init__ 构造一个 WSP 模块，然后这个 WSP 就"替换"了原来那个 Linear。
        """
        has_bias = linear.bias is not None
        return cls(
            linear.in_features,
            linear.out_features,
            has_bias,
            rank, top_k, alpha,
            linear.weight.data,
            linear.bias.data if has_bias else None,
            spectral_beta=spectral_beta,
            entropy_lam_max=entropy_lam_max,
            warmup_ratio=warmup_ratio,
            anneal_ratio=anneal_ratio,
            group_lasso_lam_max=group_lasso_lam_max,
            focus_lam_max=focus_lam_max,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 前向传播（forward）
    # ══════════════════════════════════════════════════════════════════════════
    # 整体流程（以行向量约定，即 x [B, N, d_in]）：
    #
    #   1. 冻结路径：out = x @ W₀ᵀ + b    （原始权重，不更新）
    #
    #   2. 加权输入投影：P_R(w) = I - V_k D V_kᵀ
    #      xr = x - (x @ V_k) · D · V_kᵀ
    #      效果：x 中属于 top-k 奇异方向的分量被按比例衰减
    #           （大奇异值方向衰减多 → 受保护；小奇异值方向衰减少 → 可适应）
    #
    #   3. 低秩前向：z = xr @ Aᵀ,  u = z @ Bᵀ
    #      A 把输入降到 r 维瓶颈，B 再解码到 d_out 维
    #
    #   4. 加权输出投影：P_L(w) = I - U_k D U_kᵀ
    #      ur = u - (u @ U_k) · D · U_kᵀ
    #      效果：u 中属于 top-k 奇异方向的分量被衰减
    #
    #   5. 目标显著性门控：ug = sigmoid(gate_logit) * ur
    #      每个输出通道独立学习一个 (0,1) 的门控值
    #
    #   6. 缩放残差：out += (α/r) · ug
    # ══════════════════════════════════════════════════════════════════════════

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：加权谱投影 + 显著性门控。

        参数:
            x: 输入张量 [B, N, d_in]  (B=batch, N=token数, d_in=输入维度)

        返回:
            out: 输出张量 [B, N, d_out]  (d_out=输出维度)
        """
        # ── 第 1 步：冻结路径 ──────────────────────────────────────────────
        # 普通 Linear 前向，权重 W₀ 和 bias 都是冻结的（不更新）
        out = F.linear(x, self.weight, self.bias)
        # 如果适配器未激活（rank=0），直接返回冻结路径的结果
        if not self._active or self.rank <= 0:
            return out

        # 缩放因子：α/r，控制适配器输出的整体幅度
        scale = self.alpha / self.rank

        # ── 第 2 步：加权输入投影 P_R(w) = I - V_k D V_kᵀ ────────────────
        # 从输入 x 中移除"受保护的"奇异方向分量，但移除量是加权的：
        #   大奇异值方向 → D 接近 1 → 几乎完全移除（受保护，不让 LoRA 改变）
        #   小奇异值方向 → D 接近 0 → 几乎不移除（保留适应空间）
        if self.Vk.shape[1] > 0:
            # x @ V_k: 投影到 V_k 张成的子空间  [B, N, k]
            xv = torch.matmul(x, self.Vk)
            # 乘以谱权重 D：大奇异值 × 大权重，小奇异值 × 小权重
            xv = xv * self.spectral_weight               # ← D 矩阵
            # P_R(w) · x = x - (x @ V_k) · D · V_kᵀ
            xr = x - torch.matmul(xv, self.Vk.t())
        else:
            xr = x

        # ── 第 3 步：低秩细节编码 → 目标特异性解码 ────────────────────────
        # z = xr @ Aᵀ:  输入 [B,N,d_in] → 瓶颈 [B,N,r]    ← 编码
        # u = z @ Bᵀ:   瓶颈 [B,N,r] → 输出 [B,N,d_out]   ← 解码
        z = F.linear(xr, self.lora_A)   # [B, N, r]
        u = F.linear(z, self.lora_B)    # [B, N, d_out]

        # ── 第 4 步：加权输出投影 P_L(w) = I - U_k D U_kᵀ ────────────────
        # 与输入投影同理，对 u 中属于 top-k 奇异方向的分量做加权衰减
        if self.Uk.shape[1] > 0:
            uu = torch.matmul(u, self.Uk)                # [B, N, k]
            uu = uu * self.spectral_weight               # ← D 矩阵
            ur = u - torch.matmul(uu, self.Uk.t())       # [B, N, d_out]
        else:
            ur = u

        # ── 第 5 步：目标显著性门控 ────────────────────────────────────────
        # 每个输出通道一个可学习的门控值 sigmoid(gate_logit) ∈ (0, 1)
        g = torch.sigmoid(self.gate_logit)  # [d_out]
        # 门控 × 适配器输出：重要通道保留，不重要通道被抑制
        ug = ur * g                          # [B, N, d_out]

        # ── 第 6 步：缩放残差 ──────────────────────────────────────────────
        # 最终输出 = 冻结路径 + 缩放后的门控适配器输出
        return out + scale * ug

    # ══════════════════════════════════════════════════════════════════════════
    # 正则化（Regularisation）
    # ══════════════════════════════════════════════════════════════════════════
    # 三项正则化：
    #   1. 熵正则化（Entropy）—— 推动门控值二值化（趋向 0 或 1）
    #      动机：小目标只需要少数通道响应，多数通道要彻底关闭
    #   2. Group-Lasso —— 推动 B 矩阵的行稀疏
    #      动机：神经元级压缩，与门控协同工作
    #   3. Gini 聚焦（Focus）—— 最大化门控的 Gini 系数（V2 关闭）
    #      动机：推动门控集中在极少数通道
    #
    # 三项正则化都使用 _schedule_lam() 进行 warmup + anneal 调度：
    #   Epoch 0~20:  λ=0（自由学习）
    #   Epoch 20~30: λ 线性爬坡
    #   Epoch 30~40: λ=最大值
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _gini_coefficient(g: torch.Tensor) -> torch.Tensor:
        """
        计算门控值的 Gini 系数。

        Gini 系数衡量"集中度"：0 = 均匀分布，1 = 完全集中在一个通道。
        高 Gini → 少数通道门控高，多数通道接近 0 → 稀疏的目标显著性模式。

        数学公式：
            G(g) = Σ (2i-n-1) · g_{(i)} / (n · Σ g_{(i)})
            其中 g_{(i)} 是门控值从大到小排列

        反无人机动机：
            小目标只占据图像中极少数像素，对应的特征也应该只占据少数判别通道。
            最大化 Gini 就是在推动这种稀疏集中的显著性模式。
        """
        n = g.shape[0]
        g_sorted, _ = torch.sort(g, descending=True)
        # Normalised Gini: 0 (uniform) → 1 (fully concentrated)
        weights = 2.0 * torch.arange(1, n + 1, device=g.device).float() - n - 1.0
        numerator = (weights * g_sorted).sum()
        denominator = n * g_sorted.sum() + 1e-10
        return numerator / denominator

    def _schedule_lam(self) -> Tuple[float, float, float]:
        """
        正则化权重调度：根据训练进度返回当前 epoch 的 (熵λ, group-lassoλ, 聚焦λ)。

        调度策略：
            warmup_ratio 之前 → λ=0（自由学习，无正则化压力）
            warmup~warmup+anneal → λ 线性爬坡（逐步施加正则化）
            warmup+anneal 之后 → λ=最大（全开）

        以 V2 默认参数为例（warmup_ratio=0.5, anneal_ratio=0.25, total=40 epoch）：
            Epoch  0~20:  λ=0
            Epoch 20~30:  λ 从 0 → max
            Epoch 30~40:  λ=max
        """
        p = self._global_progress
        if p < self.warmup_ratio:
            ramp = 0.0
        elif p < self.warmup_ratio + self.anneal_ratio:
            ramp = (p - self.warmup_ratio) / self.anneal_ratio
        else:
            ramp = 1.0
        return (
            self.entropy_lam_max * ramp,
            self.group_lasso_lam_max * ramp,
            self.focus_lam_max * ramp,
        )

    def regularisation_loss(self) -> torch.Tensor:
        """
        计算单个 WSP 模块的正则化损失。

        三项损失：
            L_reg = λ_e·H(g) + λ_g·mean(||B_i||₂) - λ_f·G(g)
                    熵          group-lasso (行)    聚焦(Gini↑)

        需要先调用 set_progress() 设置训练进度，再调用 forward() 前向传播，
        最后调用本方法。结果会被 collect_wsp_regularisation() 收集汇总。

        返回:
            标量张量，三项正则化损失的加权和
        """
        if not self._active:
            return torch.tensor(0.0, device=self.weight.device)

        lam_e, lam_g, lam_f = self._schedule_lam()
        loss = torch.tensor(0.0, device=self.weight.device)

        # ── 第 1 项：熵正则化 H(g) → 推动门控二值化 ──────────────────────
        # H(g) = -[g·log(g) + (1-g)·log(1-g)]
        # g=0.5 → H=log2（最大）；g→0 或 g→1 → H→0
        # 最小化 H → 门控趋向极端值（要么 0 要么 1）
        if lam_e > 0:
            g = torch.sigmoid(self.gate_logit)
            eps = 1e-8
            entropy = -(g * torch.log(g + eps) + (1.0 - g) * torch.log(1.0 - g + eps))
            loss = loss + lam_e * entropy.mean()

        # ── 第 2 项：Group-Lasso 行稀疏（B 矩阵）─────────────────────────
        # 对 B 的每一行计算 L2 范数然后求平均
        # 行范数小 → 该输出通道对应的整行都接近 0 → 通道被"剪枝"
        # 与门控协同：门控关 + B 行归零 → 彻底禁用该通道
        if lam_g > 0:
            row_norms = torch.norm(self.lora_B, p=2, dim=1)  # [d_out]
            loss = loss + lam_g * row_norms.mean()

        # ── 第 3 项：Gini 聚焦正则化（V2 关闭，FOCUS_LAM_MAX=0）─────────
        # 最大化门控的 Gini 系数 → 少数通道门控高，多数通道门控低
        # 反无人机动机：小目标只占据少数判别通道
        if lam_f > 0:
            g = torch.sigmoid(self.gate_logit)
            gini = self._gini_coefficient(g)
            loss = loss - lam_f * gini   # 负号 = 最大化 Gini

        return loss

    # ══════════════════════════════════════════════════════════════════════════
    # 合并到权重（推理用）
    # ══════════════════════════════════════════════════════════════════════════
    # 训练完成后调用，将适配器永久性地合并到原始权重中：
    #
    #   W_merged = W₀ + (α/r) · diag(g) · P_L(w) · B · A · P_R(w)
    #
    # 合并后推理时不需要再走适配器路径，结构等价于普通 Linear 层 → 零开销。
    #
    # 合并顺序（数学上等价于 forward）：
    #   1. 计算 BA = B @ A
    #   2. P_R(w) 从右边投影：BA ← BA - (BA @ V_k) · D · V_kᵀ
    #   3. P_L(w) 从左边投影：BA ← BA - U_k · D · (U_kᵀ @ BA)
    #   4. 门控缩放：ΔW = scale · diag(g) · BA
    #   5. 合并：W_merged = W₀ + ΔW
    # ══════════════════════════════════════════════════════════════════════════

    def merge_to_weight(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        将适配器合并到冻结权重中，实现推理零开销。

        合并后 WSP 模块等价于一个普通 nn.Linear，不需要额外的适配器计算。

        返回:
            (W_merged, bias_merged): 合并后的权重和偏置
        """
        if not self._active:
            return self.weight.data, self.bias.data if self.bias is not None else None

        scale = self.alpha / self.rank
        g = torch.sigmoid(self.gate_logit)  # [d_out]，门控值也合并进去

        # 先把 B @ A 合在一起 [d_out, d_in]
        BA = self.lora_B @ self.lora_A

        # 从右边施加加权投影 P_R(w)：
        #   BA ← BA - (BA @ V_k) · D · V_kᵀ
        # 相当于：BA 中属于 V_k 张成子空间的分量被衰减
        if self.Vk.shape[1] > 0:
            BA_Vk = BA @ self.Vk                       # [d_out, k]
            BA_Vk = BA_Vk * self.spectral_weight       # ← D 矩阵
            BA = BA - BA_Vk @ self.Vk.t()               # [d_out, d_in]

        # 从左边施加加权投影 P_L(w)：
        #   BA ← BA - U_k · D · (U_kᵀ @ BA)
        if self.Uk.shape[1] > 0:
            UkT_BA = self.Uk.t() @ BA                   # [k, d_in]
            UkT_BA = self.spectral_weight.unsqueeze(1) * UkT_BA  # ← D 矩阵
            BA = BA - self.Uk @ UkT_BA                   # [d_out, d_in]

        # 门控缩放：每个输出通道独立缩放
        delta = scale * g.unsqueeze(1) * BA   # [d_out, d_in]
        W_merged = self.weight.data + delta
        bias_merged = self.bias.data if self.bias is not None else None
        return W_merged, bias_merged


# ──────────────────────────────────────────────────────────────────────────────
# 顶层函数 1：收集模型所有 WSP 模块的正则化损失
# ──────────────────────────────────────────────────────────────────────────────
# 遍历模型的所有子模块（递归），找到所有 WeightedSpectralProjection 实例，
# 累加它们的 regularisation_loss() 结果。
#
# 在 OSTrackActor 中调用：
#   loss = task_loss + collect_wsp_regularisation(model)
# ──────────────────────────────────────────────────────────────────────────────

def collect_wsp_regularisation(model: nn.Module) -> torch.Tensor:
    """
    收集整个模型中所有 WSP 模块的正则化损失并求和。

    在前向传播之后调用，把返回的正则化损失加到任务损失上：
        loss = task_loss + collect_wsp_regularisation(model)

    参数:
        model: 包含 WSP 模块的 PyTorch 模型（如 OSTrack）

    返回:
        标量张量，所有 WSP 模块正则化损失的总和
    """
    total = None
    for m in model.modules():
        if isinstance(m, WeightedSpectralProjection):
            if total is None:
                total = m.regularisation_loss()
            else:
                total = total + m.regularisation_loss()
    if total is None:
        # 没有找到任何 WSP 模块，返回 0
        first_param = next(model.parameters())
        return torch.tensor(0.0, device=first_param.device, dtype=first_param.dtype)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# 顶层函数 2：注入 WSP 适配器到 ViT Backbone
# ──────────────────────────────────────────────────────────────────────────────
# 由 build_ostrack() 在 ostrack.py 中调用。
#
# 工作流程：
#   1. 遍历 layer_configs 中的每个配置组
#   2. 对每个组，遍历指定的 blocks 和 targets
#   3. 调用 _get_linear() 查找目标 nn.Linear 层
#   4. 调用 WeightedSpectralProjection.from_linear() 替换它
#
# layer_configs 每个配置组可以指定：
#   - blocks: [block索引列表] 哪些 Transformer block 要注入
#   - targets: ["attn.qkv", ...] 目标层的名字
#   - rank, top_k, alpha, spectral_beta: 该组统一的超参数
# ──────────────────────────────────────────────────────────────────────────────

def inject_wsp_into_backbone(
    backbone: nn.Module,
    enable: bool,
    layer_configs: Optional[List[Dict[str, Any]]] = None,
    entropy_lam_max: float = 1e-4,
    warmup_ratio: float = 0.33,
    anneal_ratio: float = 0.33,
    group_lasso_lam_max: float = 1e-5,
    focus_lam_max: float = 1e-5,
) -> Tuple[int, int]:
    """
    将 WeightedSpectralProjection 适配器注入 ViT backbone。

    每个配置组指定哪些 block、哪些 Linear 层、用什么超参数。

    参数:
        backbone: VisionTransformer 或 VisionTransformerCE 实例
        enable: 总开关（False → 什么都不做）
        layer_configs: 每组配置（见 WSP_DEFAULT_PRIOR_CONFIG）
        entropy_lam_max: 熵正则化峰值强度
        warmup_ratio: 正则化预热比例
        anneal_ratio: 正则化爬坡比例
        group_lasso_lam_max: Group-Lasso 峰值强度
        focus_lam_max: Gini 聚焦峰值强度

    返回:
        (num_replaced, num_frozen_blocks): 替换的 Linear 数量和冻结的 block 数
    """
    if not enable:
        return 0, 0

    configs = layer_configs or WSP_DEFAULT_PRIOR_CONFIG
    replaced = 0
    num_blocks = len(backbone.blocks)

    for group_cfg in configs:
        rank = int(group_cfg["rank"])
        top_k = int(group_cfg["top_k"])
        alpha = float(group_cfg["alpha"])
        spectral_beta = float(group_cfg.get("spectral_beta", 1.0))
        targets: Sequence[str] = tuple(group_cfg["targets"])

        for blk_idx in group_cfg["blocks"]:
            if blk_idx < 0 or blk_idx >= num_blocks:
                continue
            block = backbone.blocks[blk_idx]
            for target_name in targets:
                # 在 block 里找到指定名字的 Linear 层
                linear, parent_mod, leaf_name = _get_linear(block, target_name)
                if linear is not None:
                    # 替换：把原来的 Linear 替换成 WSP 模块
                    setattr(
                        parent_mod,
                        leaf_name,
                        WeightedSpectralProjection.from_linear(
                            linear,
                            rank=rank,
                            top_k=top_k,
                            alpha=alpha,
                            spectral_beta=spectral_beta,
                            entropy_lam_max=entropy_lam_max,
                            warmup_ratio=warmup_ratio,
                            anneal_ratio=anneal_ratio,
                            group_lasso_lam_max=group_lasso_lam_max,
                            focus_lam_max=focus_lam_max,
                        ),
                    )
                    replaced += 1

    frozen_blocks = _count_frozen(backbone, configs)
    return replaced, frozen_blocks


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具函数 1：根据名字在模块中查找 nn.Linear 层
# ──────────────────────────────────────────────────────────────────────────────
# 支持用点号分隔的嵌套名字，例如：
#   "attn.qkv"  → 先找 mod.attn，再找 mod.attn.qkv
#   "mlp.fc1"   → 先找 mod.mlp，再找 mod.mlp.fc1
# 只返回 nn.Linear 类型的层，其他类型不匹配。
# ──────────────────────────────────────────────────────────────────────────────

def _get_linear(
    parent: nn.Module, name: str
) -> Tuple[Optional[nn.Linear], Optional[nn.Module], str]:
    """
    通过名字在模块中查找 nn.Linear 层。

    支持嵌套名字（如 "attn.qkv"），逐级查找。

    返回:
        (linear, parent_mod, attr_name): (找到的Linear层, 父模块, 属性名)
        如果没找到或类型不是 nn.Linear，返回 (None, None, "")
    """
    parts = name.split(".")
    mod = parent
    for p in parts[:-1]:
        mod = getattr(mod, p, None)
        if mod is None:
            return None, None, ""
    leaf = getattr(mod, parts[-1], None)
    if isinstance(leaf, (nn.Linear,)):
        return leaf, mod, parts[-1]
    return None, None, ""


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具函数 2：统计哪些 block 没有被注入适配器（冻结的）
# ──────────────────────────────────────────────────────────────────────────────
# 在 inject_wsp_into_backbone 的最后调用，用于日志打印。
# 所有未出现在任何配置组中的 block 都被视为冻结。
# ──────────────────────────────────────────────────────────────────────────────

def _count_frozen(backbone: nn.Module, configs: List[dict]) -> int:
    """
    统计没有被注入适配器的 block 数量（即冻结的 block）。

    逻辑：收集所有配置组中出现的 block 索引，然后从 0~num_blocks-1
    中找出没有出现的那些索引。
    """
    all_injected = set()
    for g in configs:
        all_injected.update(g["blocks"])
    frozen = [i for i in range(len(backbone.blocks)) if i not in all_injected]
    if frozen:
        import logging
        _log = logging.getLogger(__name__)
        _log.info(f"WSP: blocks {frozen} are FROZEN (no adapter injected)")
    return len(frozen)
