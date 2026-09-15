"""
ASC-LoRA — Anti-UAV Small-Target Adapter.

A parameter-efficient fine-tuning method designed specifically for the
challenges of anti-UAV tracking (small targets, fast motion, frequent
occlusion). The core mathematical innovation replaces the hard orthogonal
projection in prior PEFT methods with a **spectrally weighted** projection:

    ΔW = (α/r) · diag(g) · P_L(w) · B · A · P_R(w)

    where  P_R(w) = I - V_k D V_k^T     (weighted input projection)
           P_L(w) = I - U_k D U_k^T     (weighted output projection)
           D = diag(d_1, ..., d_k)      (spectral weight matrix)
           d_i = (σ_i / σ_1)^β          (β ≥ 0, defaults to 1.0)

Motivation — why weighted projection matters for small targets:
─────────────────────────────────────────────────────────────────
Prior methods (OPLoRA, LoRA) use a HARD orthogonal projection that
completely removes the top-k singular directions from the adaptation
path. This is problematic for small-target tracking because:

  1. Small-target information (edge details, fine texture, high-frequency
     boundary responses) is disproportionately carried by the MID-RANKED
     singular directions — exactly those that hard projection weakens.
  2. Large singular values encode coarse, global structure (background
     scene, large-object patterns) that is less relevant for UAV targets.
  3. A hard cut (d_i ∈ {0,1}) discards all top-k subspace information
     equally, when in reality σ_k holds far more "detail budget" than σ₁.

ASC-LoRA replaces hard cut with GRADUATED protection: σ₁ direction is nearly
fully protected (d₁=1), while σ_k direction retains most adaptation freedom
(d_k = (σ_k/σ₁)^β), preserving a continuous gradient of detail sensitivity.

Architecture overview:
─────────────────────
  Frozen path:        y_base = x @ W₀ᵀ + b
  Adapter path:
    x_r = P_R(w) · x          — weighted input projection
    z   = x_r · Aᵀ            — low-rank detail encoding  (r-dim bottleneck)
    u   = z · Bᵀ              — target-specific decoding   (d_out-dim)
    u_r = P_L(w) · u          — weighted output projection
    u_g = σ(s) · u_r          — target-saliency gate (learnable per neuron)
    Δy  = (α/r) · u_g         — scaled residual

Regularisation:
  L_reg = λ_e·H(σ(s)) + λ_g·Σ||B_i||₂ - λ_f·G(σ(s))
         ↑ entropy        ↑ group-lasso     ↑ focus (Gini↑)
           (sparsify)     (neuron prune)     (concentrate)

Design lineage — key differences from prior PEFT:
─────────────────────────────────────────────────
  vs. LoRA           — adds spectral-gated orthogonal projection
  vs. OPLoRA         — weighted (not hard) projection + learnable gates
  vs. NS-OPLoRA      — no static binary mask; fully differentiable
  vs. SGLoRA (prior) — weighted UDUᵀ (not UUᵀ) + focus reg + detail init
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

def compute_spectral_weights(
    S: torch.Tensor,
    k: int,
    beta: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute spectral attenuation coefficients for weighted projection.

    Args:
        S: singular values of W₀  [min(d_out, d_in)]
        k: number of top components to weight
        beta: exponent controlling attenuation profile.
            β=0  → all d_i=1 → reverts to original hard projection P = I - UUᵀ
            β=1  → d_i = σ_i/σ₁ → linear attenuation (default)
            β>1  → faster decay → more protection for top components
            β<1  → slower decay → more adaptation freedom for all components
        eps: numerical stability

    Returns:
        d: spectral weights  [k]  in [0, 1], d₁ = 1, d_k = (σ_k/σ₁)^β
    """
    sigma_1 = S[0]
    d = (S[:k] / (sigma_1 + eps)).pow(beta).clamp(0.0, 1.0)
    return d


def _spectral_weights_for_mode(
    S: torch.Tensor, k: int, beta: float, mode: str
) -> torch.Tensor:
    """Return the single source of truth for all mode-dependent d_i values."""
    if mode == "hard":
        return torch.ones(k, device=S.device, dtype=S.dtype)
    if mode == "weighted":
        return compute_spectral_weights(S, k, beta)
    if mode == "none":
        # SVD is deliberately still retained for gate initialisation and the
        # complement constraint, but d is not used by the forward projection.
        return compute_spectral_weights(S, k, beta)
    raise ValueError("SPECTRAL_PROJECTION must be one of weighted, hard, none; "
                     f"got {mode!r}")


def _detail_saliency_from_svd(
    Uk: torch.Tensor,
    S: torch.Tensor,
    k: int,
    beta: float = 1.0,
    spectral_projection: str = "weighted",
) -> torch.Tensor:
    """
    Compute per-output-channel detail-saliency bias for gate initialisation.

    The insight: output channels that strongly participate in small-σ
    (fine-detail) singular directions should start with slightly higher
    gate values, giving small-target features a head start in training.

    Returns:
        bias: per-channel bias  [d_out]  in [-0.3, 0.3]
    """
    d = _spectral_weights_for_mode(S, k, beta, spectral_projection)
    # Each output channel's "detail weight" = sum over singular directions
    # of (Uk[i,:]² · (1 - d)), so channels coupled to small-σ get higher weight
    detail = (Uk.pow(2) * (1.0 - d).unsqueeze(0)).sum(dim=1)  # [d_out]
    # Normalise to [-0.3, 0.3] range — small bias, won't overwhelm task loss
    d_max = detail.max()
    if d_max > 1e-10:
        detail = 0.6 * detail / d_max - 0.3
    else:
        detail = torch.zeros_like(detail)
    return detail


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
# ---------------------------------------------------------------------------

ASC_LORA_DEFAULT_PRIOR_CONFIG: List[Dict[str, Any]] = [
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


# ---------------------------------------------------------------------------
# ASCLoRA  (core PEFT module)
# ---------------------------------------------------------------------------

class ASCLoRA(nn.Module):
    """
    Single ASC-LoRA adapter module — the building block of anti-UAV PEFT.

    Replaces one nn.Linear in the ViT backbone. The frozen weight W₀ is
    preserved; a low-rank adapter with spectrally weighted projections
    and learnable gates provides the parameter-efficient update.

    Key parameters:
        spectral_beta: exponent controlling how gradually the projection
            attenuates top singular directions. β=1 is a good default;
            lower values (0.3-0.7) give more adaptation freedom at the
            cost of less protection; higher values (1.5-2.0) are more
            conservative.
        focus_lam_max: peak strength of the focus (Gini) regularisation.
            Push gates toward sparse concentration — useful when the
            tracker must learn highly selective features for small targets.

    Forward (row-vector convention):
        out  = x @ W0^T + bias                       … frozen base
        x_r  = x - (x @ V_k) · D · V_k^T             … weighted input projection
        z    = x_r @ A^T                               … low-rank detail encoding
        u    = z @ B^T                                 … target-specific decoding
        u_r  = u - (u @ U_k) · D · U_k^T              … weighted output projection
        u_g  = sigmoid(s) * u_r                        … target-saliency gate
        out += (alpha / rank) * u_g                    … task-adapted residual
    """

    # ---- class-level progress tracking ------------------------------------
    _global_progress: float = 0.0  # [0, 1]  current_epoch / total_epochs

    @classmethod
    def set_progress(cls, progress: float) -> None:
        """Set training progress ratio [0, 1] for all ASC-LoRA instances."""
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
        # Spectral weighting  (anti-UAV innovation)
        spectral_beta: float = 1.0,
        # Entropy schedule
        entropy_lam_max: float = 1e-4,
        warmup_ratio: float = 0.33,
        anneal_ratio: float = 0.33,
        # Group-lasso on B rows
        group_lasso_lam_max: float = 1e-5,
        # Focus regularisation  (anti-UAV: sparse target-saliency concentration)
        focus_lam_max: float = 1e-5,
        channel_gate: bool = True,
        spectral_projection: str = "weighted",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.top_k = int(top_k)
        self.spectral_beta = float(spectral_beta)
        self.channel_gate = bool(channel_gate)
        self.spectral_projection = str(spectral_projection).lower()
        if self.spectral_projection not in {"weighted", "hard", "none"}:
            raise ValueError("spectral_projection must be 'weighted', 'hard', or 'none'")
        self.dropout = nn.Dropout(float(dropout))

        # Regularisation schedule
        self.entropy_lam_max = float(entropy_lam_max)
        self.warmup_ratio = float(warmup_ratio)
        self.anneal_ratio = float(anneal_ratio)
        self.group_lasso_lam_max = float(group_lasso_lam_max)
        self.focus_lam_max = float(focus_lam_max)

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
            self.register_parameter("gate_logit", None)
            return

        self._active = True

        # ---- SVD: extract U_k, V_k for weighted projections ----------------
        W = self.weight.data
        kk = min(top_k, min(out_features, in_features))
        if kk > 0:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            kk = min(kk, U.shape[1], Vh.shape[0])
            Uk = U[:, :kk]            # [d_out, k]
            Vk = Vh[:kk, :].T          # [d_in, k]
            sigma = S[:kk]             # [k]

            # ---- Weighted Spectral Projection: D = diag(d) ------------------
            # d_i = (σ_i / σ₁)^β  ∈ [0, 1]
            # This is the KEY anti-UAV innovation: soft graduated protection
            # instead of hard cut. Small-target detail adapts through mid-rank σ.
            d = _spectral_weights_for_mode(
                sigma, kk, self.spectral_beta, self.spectral_projection
            )

            # ---- Detail-saliency gate initialisation -------------------------
            # Output channels aligned with small-σ singular directions get a
            # slight positive bias, giving small-target features an early start.
            gate_bias = _detail_saliency_from_svd(
                Uk, sigma, kk, self.spectral_beta, self.spectral_projection
            )
            # Base starts at 0 (sigmoid=0.5), plus detail bias
            gate_logit_init = gate_bias
        else:
            Uk = torch.empty(out_features, 0, device=W.device, dtype=W.dtype)
            Vk = torch.empty(in_features, 0, device=W.device, dtype=W.dtype)
            d = torch.empty(0, device=W.device, dtype=W.dtype)
            gate_logit_init = torch.zeros(out_features, device=W.device, dtype=W.dtype)

        self.register_buffer("spectral_weight", d.contiguous())   # [k]
        self.register_buffer("Uk", Uk.contiguous())                # [d_out, k]
        self.register_buffer("Vk", Vk.contiguous())                # [d_in, k]

        # ---- Low-rank adapter matrices ------------------------------------
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # ---- Target-saliency gate logits ---------------------------------
        # Gate-off is a true parameter ablation: do not allocate or optimise s.
        self.register_parameter(
            "gate_logit", nn.Parameter(gate_logit_init) if self.channel_gate else None
        )

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
        spectral_beta: float = 1.0,
        entropy_lam_max: float = 1e-4,
        warmup_ratio: float = 0.33,
        anneal_ratio: float = 0.33,
        group_lasso_lam_max: float = 1e-5,
        focus_lam_max: float = 1e-5,
        channel_gate: bool = True,
        spectral_projection: str = "weighted",
        dropout: float = 0.0,
    ) -> "ASCLoRA":
        """Create an ASC-LoRA adapter from an existing nn.Linear."""
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
            spectral_beta=spectral_beta,
            entropy_lam_max=entropy_lam_max,
            warmup_ratio=warmup_ratio,
            anneal_ratio=anneal_ratio,
            group_lasso_lam_max=group_lasso_lam_max,
            focus_lam_max=focus_lam_max,
            channel_gate=channel_gate,
            spectral_projection=spectral_projection,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted spectral projection + saliency gate."""
        out = F.linear(x, self.weight, self.bias)
        if not self._active or self.rank <= 0:
            return out

        scale = self.alpha / self.rank

        # Apply both spectral projections in low-rank space.  Materialising
        # x @ P_R and (z @ B^T) @ P_L would create several [B, N, d] tensors
        # per adapter and causes OOM at the A4 batch size.  The expressions
        # below are algebraically identical to P_L B A P_R, but only retain
        # [B, N, rank] and [B, N, top_k] intermediates until the final update.

        # --- Weighted input projection folded into lora_A -------------------
        adapter_input = self.dropout(x)
        z = F.linear(adapter_input, self.lora_A)  # x @ A^T, [B, N, r]
        if self.spectral_projection != "none" and self.Vk.shape[1] > 0:
            xv = torch.matmul(adapter_input, self.Vk)               # [B, N, k]
            right_correction = self.Vk.t() @ self.lora_A.t()  # [k, r]
            z = z - torch.matmul(xv * self.spectral_weight, right_correction)

        # --- Weighted output projection folded into lora_B ------------------
        lora_B = self.lora_B
        if self.spectral_projection != "none" and self.Uk.shape[1] > 0:
            left_correction = self.Uk.t() @ lora_B       # [k, r]
            lora_B = lora_B - self.Uk @ (
                self.spectral_weight.unsqueeze(1) * left_correction
            )

        # Fold the channel gate into the projected low-rank decoder as well.
        gated_lora_B = lora_B
        if self.channel_gate:
            gated_lora_B = torch.sigmoid(self.gate_logit).unsqueeze(1) * lora_B

        return out + scale * F.linear(z, gated_lora_B)

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    @staticmethod
    def _gini_coefficient(g: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gini coefficient of gate values.

        Higher Gini → activation concentrated in few channels → sparse
        target-saliency pattern.  Used in focus regularisation to push
        the adapter toward selective, small-target-like features.
        """
        n = g.shape[0]
        g_sorted, _ = torch.sort(g, descending=True)
        # Normalised Gini: 0 (uniform) → 1 (fully concentrated)
        weights = 2.0 * torch.arange(1, n + 1, device=g.device).float() - n - 1.0
        numerator = (weights * g_sorted).sum()
        denominator = n * g_sorted.sum() + 1e-10
        return numerator / denominator

    def _schedule_lam(self) -> Tuple[float, float, float]:
        """Return (entropy_lam, group_lasso_lam, focus_lam) at current progress."""
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
        Compute per-module regularisation loss.

        L_reg = λ_e·H(g) + λ_g·mean(||B_{i,:}||₂) - λ_f·G(g)
                entropy        group-lasso B rows       focus (Gini↑)

        Call after forward(); progress should be set via set_progress().
        """
        if not self._active:
            return torch.tensor(0.0, device=self.weight.device)

        lam_e, lam_g, lam_f = self._schedule_lam()
        loss = torch.tensor(0.0, device=self.weight.device)

        # Entropy regularisation: H(g) → push gates toward 0 or 1
        if self.channel_gate and lam_e > 0:
            g = torch.sigmoid(self.gate_logit)
            eps = 1e-8
            entropy = -(g * torch.log(g + eps) + (1.0 - g) * torch.log(1.0 - g + eps))
            loss = loss + lam_e * entropy.mean()

        # Group-lasso on rows of B: neuron-level sparsity
        if lam_g > 0:
            row_norms = torch.norm(self.lora_B, p=2, dim=1)  # [d_out]
            loss = loss + lam_g * row_norms.mean()

        # Focus regularisation: maximise Gini → sparse gate concentration
        # This is anti-UAV specific: small targets need few selective channels
        if self.channel_gate and lam_f > 0:
            g = torch.sigmoid(self.gate_logit)
            gini = self._gini_coefficient(g)
            loss = loss - lam_f * gini  # negative sign = maximise Gini

        return loss

    def complement_constraint_loss(self) -> torch.Tensor:
        """Raw complement constraint, independent from its training weight.

        It penalises the protected spectral components of BA.  The same d_i
        drives every mode: weighted d_i, hard all-ones, and (for F2) weighted
        d_i even though the forward projection itself is disabled.
        """
        if not self._active or self.Uk.shape[1] == 0:
            return self.weight.new_zeros(())
        BA = self.lora_B @ self.lora_A
        left = self.spectral_weight.unsqueeze(1) * (self.Uk.t() @ BA)
        right = (BA @ self.Vk) * self.spectral_weight.unsqueeze(0)
        return 0.5 * (left.pow(2).mean() + right.pow(2).mean())

    # ------------------------------------------------------------------
    # Merge (inference)
    # ------------------------------------------------------------------

    def merge_to_weight(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Merge the adapter into the frozen weight for zero-overhead inference.

        W_merged = W₀ + (α/r) · diag(g) · P_L(w) · B · A · P_R(w)
        """
        if not self._active:
            return self.weight.data, self.bias.data if self.bias is not None else None

        scale = self.alpha / self.rank
        g = torch.sigmoid(self.gate_logit) if self.channel_gate else torch.ones_like(self.weight[:, 0])

        BA = self.lora_B @ self.lora_A      # [d_out, d_in]

        # Apply P_R(w) on the right: B A - (B A @ V_k) · D · V_k^T
        if self.spectral_projection != "none" and self.Vk.shape[1] > 0:
            BA_Vk = BA @ self.Vk             # [d_out, k]
            BA_Vk = BA_Vk * self.spectral_weight  # [d_out, k]  ← weighted
            BA = BA - BA_Vk @ self.Vk.t()    # [d_out, d_in]

        # Apply P_L(w) on the left: B A - U_k · D · (U_k^T @ B A)
        if self.spectral_projection != "none" and self.Uk.shape[1] > 0:
            UkT_BA = self.Uk.t() @ BA        # [k, d_in]
            UkT_BA = self.spectral_weight.unsqueeze(1) * UkT_BA  # [k, d_in]  ← weighted
            BA = BA - self.Uk @ UkT_BA       # [d_out, d_in]

        delta = scale * g.unsqueeze(1) * BA   # [d_out, d_in]
        W_merged = self.weight.data + delta
        bias_merged = self.bias.data if self.bias is not None else None
        return W_merged, bias_merged


# ---------------------------------------------------------------------------
# Collect regularisation across the model
# ---------------------------------------------------------------------------

def collect_asc_lora_regularisation(model: nn.Module) -> torch.Tensor:
    """
    Sum regularisation_loss() over all ASC-LoRA modules.

    Call this after forward and add the result to the task loss:
        loss = task_loss + collect_asc_lora_regularisation(model)
    """
    total = None
    for m in model.modules():
        if isinstance(m, ASCLoRA):
            if total is None:
                total = m.regularisation_loss()
            else:
                total = total + m.regularisation_loss()
    if total is None:
        # 找一个 GPU 参数作为 device 锚点，避免跨设备同步
        first_param = next(model.parameters())
        return torch.tensor(0.0, device=first_param.device, dtype=first_param.dtype)
    return total


def collect_asc_lora_complement_loss(model: nn.Module) -> torch.Tensor:
    """Sum the *unweighted* complement constraint for logging and ablations."""
    total = None
    for m in model.modules():
        if isinstance(m, ASCLoRA):
            value = m.complement_constraint_loss()
            total = value if total is None else total + value
    if total is None:
        first_param = next(model.parameters())
        return torch.zeros((), device=first_param.device, dtype=first_param.dtype)
    return total


# ---------------------------------------------------------------------------
# Injection into ViT backbone
# ---------------------------------------------------------------------------

def inject_asc_lora_into_backbone(
    backbone: nn.Module,
    enable: bool,
    layer_configs: Optional[List[Dict[str, Any]]] = None,
    entropy_lam_max: float = 1e-4,
    warmup_ratio: float = 0.33,
    anneal_ratio: float = 0.33,
    group_lasso_lam_max: float = 1e-5,
    focus_lam_max: float = 1e-5,
    channel_gate: bool = True,
    spectral_projection: str = "weighted",
    dropout: float = 0.0,
) -> Tuple[int, int]:
    """
    Inject ASC-LoRA adapters into the ViT backbone.

    Each config group specifies which blocks, which target Linear layers,
    and what hyperparameters (rank, top_k, alpha, spectral_beta) to use.

    Args:
        backbone: VisionTransformer or VisionTransformerCE instance.
        enable: master switch.
        layer_configs: per-group configs (see ASC_LORA_DEFAULT_PRIOR_CONFIG).
        entropy_lam_max: peak entropy regularisation strength.
        warmup_ratio: fraction of training where λ=0.
        anneal_ratio: fraction over which λ ramps to max.
        group_lasso_lam_max: peak group-lasso regularisation strength.
        focus_lam_max: peak focus (Gini) regularisation strength.

    Returns:
        (num_replaced, num_blocks_without_asc_lora)
    """
    if not enable:
        return 0, 0

    configs = layer_configs or ASC_LORA_DEFAULT_PRIOR_CONFIG
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
                linear, parent_mod, leaf_name = _get_linear(block, target_name)
                if linear is not None:
                    setattr(
                        parent_mod,
                        leaf_name,
                        ASCLoRA.from_linear(
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
                            channel_gate=channel_gate,
                            spectral_projection=spectral_projection,
                            dropout=dropout,
                        ),
                    )
                    replaced += 1

    unadapted_blocks = _count_unadapted(backbone, configs)
    return replaced, unadapted_blocks


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
    if isinstance(leaf, (nn.Linear,)):
        return leaf, mod, parts[-1]
    return None, None, ""


def _count_unadapted(backbone: nn.Module, configs: List[dict]) -> int:
    all_injected = set()
    for g in configs:
        all_injected.update(g["blocks"])
    unadapted = [i for i in range(len(backbone.blocks)) if i not in all_injected]
    if unadapted:
        import logging
        _log = logging.getLogger(__name__)
        _log.info(f"ASC-LoRA: blocks {unadapted} have no adapter and use standard fine-tuning")
    return len(unadapted)
