"""
Spectral-Gated Low-Rank Adaptation (SGLoRA).

A deeply fused parameter-efficient fine-tuning method that replaces the
independent OPLoRA + NeuroAda stack with a single unified formulation:

    ΔW = (alpha / r) · diag(g) · P_L · B · A · P_R

    where  P_R = I - V_k V_k^T,   P_L = I - U_k U_k^T

Design principles (vs. NS-OPLoRA):
  1. BOTH projections P_R and P_L are RETAINED — the full OPLoRA orthogonal
     skeleton protects pre-trained principal components in input AND output space.
  2. The static binary neuron mask M ∈ {0,1}^{d_out} (NeuroAda) is REPLACED by
     a learnable soft spectral gate g = sigmoid(s) ∈ [0,1]^{d_out}, placed AFTER
     P_L so it can selectively scale each neuron's projected update.
  3. The gate g is initialised from the SAME SVD spectrum (spectral importance)
     and trained with an entropy-regularisation schedule that gradually pushes
     gates toward {0,1} after a warm-up period.
  4. Row-wise group-lasso on B provides complementary neuron-level sparsity.

Reference:
  Xiong & Xie (AAAI-26)  — OPLoRA orthogonal projection
  Zhang et al. (EMNLP 25) — NeuroAda neuron selectivity
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Default layer config — evidence-based, anti-UAV tracking
#
# Derived from perturbation experiments (OSTrack_Layer_Analysis.md).
# Comprehensive sensitivity = PatchShuffle + GaussBlur + PhaseScramble:
#
#   B6: 1.559  (highest)  — CE comprehensive decision, GaussBlur peak
#   B7: 1.491             — fusion hub, texture+structure dual-track
#   B9: 1.479             — CE texture screening, PatchShuffle peak
#   B5: 1.380             — post-CE reconstruction
#   B1: 1.224             — texture ignition (4.6× jump from B0)
#   B3: 1.170             — CE spatial screening
#   B2: 1.061             — global spatial construction
#   ★ B0, B4, B8, B10, B11 are FROZEN (low sensitivity or no learning capacity)
#
# Key correction vs. original NS-OPLoRA heuristic:
#   B7 was wrongly frozen — it is actually the 2nd most important layer.
# ---------------------------------------------------------------------------

SGLORA_DEFAULT_LAYER_CONFIG: List[Dict[str, Any]] = [
    # Group A: core adaptation — highest-sensitivity layers
    #   B7: fusion hub (comprehensive=1.491), texture+structure dual-track
    #   B1: texture ignition (comprehensive=1.224), 4.6× PatchShuffle jump
    {
        "blocks": [7, 1],
        "rank": 8,
        "top_k": 16,
        "alpha": 8.0,
        "targets": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
    },
    # Group B: CE decision layers — rank=8, attention-focused
    #   B6: comprehensive decision (comprehensive=1.559, GaussBlur peak 0.538)
    #   B9: texture screening (comprehensive=1.479, PatchShuffle peak 0.713)
    #   B3: spatial screening (comprehensive=1.170, CE entry)
    {
        "blocks": [6, 9, 3],
        "rank": 8,
        "top_k": 16,
        "alpha": 8.0,
        "targets": ["attn.qkv", "attn.proj"],
    },
    # Group C: structural support — MLP-only medium adaptation
    #   B5: post-CE reconstruction (comprehensive=1.380, two-metric dual-high)
    #   B2: global spatial construction (comprehensive=1.061, new-view adaptation)
    {
        "blocks": [5, 2],
        "rank": 4,
        "top_k": 16,
        "alpha": 8.0,
        "targets": ["mlp.fc1", "mlp.fc2"],
    },
    # Group D: occlusion/edge-case support — minimal adaptation
    #   B8: pure local texture (comprehensive=0.843), critical for occluded targets
    {
        "blocks": [8],
        "rank": 2,
        "top_k": 16,
        "alpha": 8.0,
        "targets": ["attn.qkv", "attn.proj"],
    },
    # Frozen:
    #   B0:  spatial embedding,  no learning capacity,  comprehensive=0.729
    #   B4:  spatial refinement,  low-sensitivity gap +0.096, comprehensive=1.089
    #   B10: fine-tuning,         low priority per doc
    #   B11: stable output,       perturbation-insensitive, comprehensive=1.025
]


# ---------------------------------------------------------------------------
# SpectralGatedLinear
# ---------------------------------------------------------------------------

class SpectralGatedLinear(nn.Module):
    """
    Single unified PEFT module.

    Forward (row-vector convention):
        out  = x @ W0^T + bias                              … frozen base
        x_r  = x - (x @ V_k) @ V_k^T                        … input projection (P_R)
        z    = x_r @ A^T                                      … low-rank bottleneck (r)
        u    = z @ B^T                                        … expand to d_out
        u_r  = u - (u @ U_k) @ U_k^T                         … output projection (P_L)
        u_g  = sigmoid(s) * u_r                              … spectral gate (after P_L)
        out += (alpha / r) * u_g                             … residual

    Where:
        U_k  — top-k left singular vectors of W0   (frozen buffer)
        V_k  — top-k right singular vectors of W0  (frozen buffer)
        A    — (r × d_in)  trainable
        B    — (d_out × r) trainable
        s    — (d_out,)    trainable gate logits

    Regularisation (applied externally via .regularisation_loss()):
        L_reg = λ_e(t) · H(sigmoid(s)) + λ_g(t) · Σ_i ||B_{i,:}||_2
    """

    # ---- class-level progress tracking (set once per training step) ----------
    _global_progress: float = 0.0  # [0, 1]  current_epoch / total_epochs

    @classmethod
    def set_progress(cls, progress: float) -> None:
        """Set training progress ratio [0, 1] for all SpectralGatedLinear instances."""
        cls._global_progress = max(0.0, min(1.0, float(progress)))

    # ------------------------------------------------------------------

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        rank: int,
        top_k: int,
        alpha: float,
        weight: torch.Tensor,
        bias_tensor: Optional[torch.Tensor],
        # Entropy schedule
        entropy_lam_max: float = 1e-4,
        warmup_ratio: float = 0.33,
        anneal_ratio: float = 0.33,
        # Group-lasso
        group_lasso_lam_max: float = 1e-5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.top_k = int(top_k)

        # Regularisation schedule
        self.entropy_lam_max = float(entropy_lam_max)
        self.warmup_ratio = float(warmup_ratio)
        self.anneal_ratio = float(anneal_ratio)
        self.group_lasso_lam_max = float(group_lasso_lam_max)

        # ---- frozen weight & bias -------------------------------------------
        self.register_parameter(
            "weight", nn.Parameter(weight.detach(), requires_grad=False)
        )
        if bias and bias_tensor is not None:
            self.register_parameter(
                "bias", nn.Parameter(bias_tensor.detach(), requires_grad=False)
            )
        else:
            self.register_parameter("bias", None)

        # ---- null-rank guard ------------------------------------------------
        if rank <= 0:
            self._active = False
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

        # ---- SVD: extract U_k, V_k for orthogonal projections --------------
        W = self.weight.data
        kk = min(top_k, min(out_features, in_features))
        if kk > 0:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            kk = min(kk, U.shape[1], Vh.shape[0])
            Uk = U[:, :kk]           # [d_out, k]
            Vk = Vh[:kk, :].T         # [d_in, k]
        else:
            Uk = torch.empty(out_features, 0, device=W.device, dtype=W.dtype)
            Vk = torch.empty(in_features, 0, device=W.device, dtype=W.dtype)

        # Neutral gate init: all gates start at 0.5 (sigmoid(0) = 0.5).
        # P_L already removes principal-component directions, so the gate
        # does NOT need to suppress "important" neurons — that would be
        # double-penalising. Instead, the gate starts neutral and learns
        # neuron-level sparsity purely from the task loss + entropy schedule.
        gate_logit_init = torch.zeros(out_features, device=W.device, dtype=W.dtype)

        self.register_buffer("Uk", Uk.contiguous())
        self.register_buffer("Vk", Vk.contiguous())

        # ---- LoRA matrices ------------------------------------------------
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # ---- Learnable gate logits ----------------------------------------
        self.gate_logit = nn.Parameter(gate_logit_init)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        top_k: int,
        alpha: float,
        entropy_lam_max: float = 1e-4,
        warmup_ratio: float = 0.33,
        anneal_ratio: float = 0.33,
        group_lasso_lam_max: float = 1e-5,
    ) -> "SpectralGatedLinear":
        has_bias = linear.bias is not None
        return cls(
            linear.in_features,
            linear.out_features,
            has_bias,
            rank,
            top_k,
            alpha,
            linear.weight.data,
            linear.bias.data if has_bias else None,
            entropy_lam_max=entropy_lam_max,
            warmup_ratio=warmup_ratio,
            anneal_ratio=anneal_ratio,
            group_lasso_lam_max=group_lasso_lam_max,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if not self._active or self.rank <= 0:
            return out

        scale = self.alpha / self.rank

        # --- input orthogonal projection: P_R @ x ----------------------------
        if self.Vk.shape[1] > 0:
            xv = torch.matmul(x, self.Vk)          # [B, N, k]
            xr = x - torch.matmul(xv, self.Vk.t())  # [B, N, d_in]
        else:
            xr = x

        # --- low-rank transform in projected space ---------------------------
        z = F.linear(xr, self.lora_A)   # [B, N, r]
        u = F.linear(z, self.lora_B)    # [B, N, d_out]

        # --- output orthogonal projection: P_L @ u ---------------------------
        if self.Uk.shape[1] > 0:
            uu = torch.matmul(u, self.Uk)           # [B, N, k]
            ur = u - torch.matmul(uu, self.Uk.t())  # [B, N, d_out]
        else:
            ur = u

        # --- spectral gate (after P_L) ---------------------------------------
        g = torch.sigmoid(self.gate_logit)  # [d_out]
        ug = ur * g                          # [B, N, d_out]

        return out + scale * ug

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def _schedule_lam(self) -> Tuple[float, float]:
        """Return (entropy_lam, group_lasso_lam) for the current progress."""
        p = self._global_progress
        if p < self.warmup_ratio:
            ramp = 0.0
        elif p < self.warmup_ratio + self.anneal_ratio:
            ramp = (p - self.warmup_ratio) / self.anneal_ratio
        else:
            ramp = 1.0
        return self.entropy_lam_max * ramp, self.group_lasso_lam_max * ramp

    def regularisation_loss(self) -> torch.Tensor:
        """
        Compute per-module regularisation loss.

        L = λ_e(t) · mean(H(g)) + λ_g(t) · mean(||B_{i,:}||_2)

        Call after forward(); progress should be set via set_progress().
        """
        if not self._active:
            return torch.tensor(0.0, device=self.weight.device)

        lam_e, lam_g = self._schedule_lam()
        loss = torch.tensor(0.0, device=self.weight.device)

        # Entropy: H(g) = -g·log(g) - (1-g)·log(1-g)
        if lam_e > 0:
            g = torch.sigmoid(self.gate_logit)
            eps = 1e-8
            entropy = -(g * torch.log(g + eps) + (1.0 - g) * torch.log(1.0 - g + eps))
            loss = loss + lam_e * entropy.mean()

        # Group-lasso on rows of B
        if lam_g > 0:
            row_norms = torch.norm(self.lora_B, p=2, dim=1)  # [d_out]
            loss = loss + lam_g * row_norms.mean()

        return loss

    # ------------------------------------------------------------------
    # Merge (inference)
    # ------------------------------------------------------------------

    def merge_to_weight(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Return (W_merged, bias) that folds the adapter into a plain Linear.

        W_merged = W0 + (alpha/r) · diag(g) · P_L · B · A · P_R

        where P_L = I - U_k @ U_k^T,  P_R = I - V_k @ V_k^T
        """
        if not self._active:
            return self.weight.data, self.bias.data if self.bias is not None else None

        scale = self.alpha / self.rank
        g = torch.sigmoid(self.gate_logit)  # [d_out]

        BA = self.lora_B @ self.lora_A                      # [d_out, d_in]

        # Apply P_R on the right: BA @ P_R = BA - (BA @ V_k) @ V_k^T
        if self.Vk.shape[1] > 0:
            BA = BA - (BA @ self.Vk) @ self.Vk.t()

        # Apply P_L on the left:  P_L @ BA = BA - U_k @ (U_k^T @ BA)
        if self.Uk.shape[1] > 0:
            BA = BA - self.Uk @ (self.Uk.t() @ BA)

        delta = scale * g.unsqueeze(1) * BA                 # [d_out, d_in]
        W_merged = self.weight.data + delta
        bias_merged = self.bias.data if self.bias is not None else None
        return W_merged, bias_merged


# ---------------------------------------------------------------------------
# Collect regularisation across the model
# ---------------------------------------------------------------------------

def collect_sglora_regularisation(model: nn.Module) -> torch.Tensor:
    """
    Sum regularisation_loss() over all SpectralGatedLinear modules in `model`.
    Call this after forward and add the result to the task loss.
    """
    total = torch.tensor(0.0)
    for m in model.modules():
        if isinstance(m, SpectralGatedLinear):
            total = total + m.regularisation_loss()
    return total


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def inject_sglora_into_backbone(
    backbone: nn.Module,
    enable: bool,
    layer_configs: Optional[List[Dict[str, Any]]] = None,
    entropy_lam_max: float = 1e-4,
    warmup_ratio: float = 0.33,
    anneal_ratio: float = 0.33,
    group_lasso_lam_max: float = 1e-5,
) -> Tuple[int, int]:
    """
    Inject SpectralGatedLinear into the ViT backbone according to layer_configs.

    Args:
        backbone: VisionTransformer or VisionTransformerCE instance.
        enable: master switch.
        layer_configs: per-group configs (see SGLORA_DEFAULT_LAYER_CONFIG).
        entropy_lam_max: peak entropy regularisation strength.
        warmup_ratio: fraction of training where λ=0.
        anneal_ratio: fraction over which λ ramps to max.
        group_lasso_lam_max: peak group-lasso regularisation strength.

    Returns:
        (num_replaced, num_frozen_blocks)
    """
    if not enable:
        return 0, 0

    configs = layer_configs or SGLORA_DEFAULT_LAYER_CONFIG
    replaced = 0
    num_blocks = len(backbone.blocks)

    for group_cfg in configs:
        rank = int(group_cfg["rank"])
        top_k = int(group_cfg["top_k"])
        alpha = float(group_cfg["alpha"])
        targets: Sequence[str] = tuple(group_cfg["targets"])

        for blk_idx in group_cfg["blocks"]:
            if blk_idx < 0 or blk_idx >= num_blocks:
                continue
            block = backbone.blocks[blk_idx]
            for target_name in targets:
                linear, parent_mod, leaf_name = _get_linear(block, target_name)
                if linear is not None:
                    setattr(
                        parent_mod,
                        leaf_name,
                        SpectralGatedLinear.from_linear(
                            linear,
                            rank=rank,
                            top_k=top_k,
                            alpha=alpha,
                            entropy_lam_max=entropy_lam_max,
                            warmup_ratio=warmup_ratio,
                            anneal_ratio=anneal_ratio,
                            group_lasso_lam_max=group_lasso_lam_max,
                        ),
                    )
                    replaced += 1

    frozen_blocks = _count_frozen(backbone, configs)
    return replaced, frozen_blocks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_linear(
    parent: nn.Module, name: str
) -> Tuple[Optional[nn.Linear], Optional[nn.Module], str]:
    """Walk nested attrs (e.g. 'attn.qkv') and return (linear, owner, attr_name)."""
    parts = name.split(".")
    mod = parent
    for p in parts[:-1]:
        mod = getattr(mod, p, None)
        if mod is None:
            return None, None, ""
    leaf = getattr(mod, parts[-1], None)
    if isinstance(leaf, nn.Linear):
        return leaf, mod, parts[-1]
    return None, None, ""


def _count_frozen(backbone: nn.Module, configs: List[dict]) -> int:
    all_injected = set()
    for g in configs:
        all_injected.update(g["blocks"])
    frozen = [i for i in range(len(backbone.blocks)) if i not in all_injected]
    if frozen:
        import logging
        _log = logging.getLogger(__name__)
        _log.info(f"SGLoRA: blocks {frozen} are FROZEN (no PEFT injected)")
    return len(frozen)
