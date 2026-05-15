"""
Orthogonal Projection LoRA (OPLoRA) for parameter-efficient fine-tuning.

Xiong & Xie (AAAI-26): ΔW = P_L B A P_R with P_L = I - U_k U_k^T, P_R = I - V_k V_k^T,
where U_k, V_k are top-k left/right singular vectors of frozen W0. Forward uses the
efficient form (no dense P_L/P_R): x -> P_R x -> A -> B -> P_L output.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OPLoRALinear(nn.Module):
    """
    y = x W0^T + b + (alpha/r) * P_L (B (A (P_R x^T)^T)^T ... in row layout:
    y += scale * ( u - (u @ U_k) @ U_k^T ) with u = (x @ A^T) @ B^T, x_r = x - (x @ V_k) @ V_k^T
    """

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
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        self.alpha = float(alpha)

        self.register_parameter(
            "weight",
            nn.Parameter(weight.detach(), requires_grad=False),
        )
        if bias and bias_tensor is not None:
            self.register_parameter(
                "bias",
                nn.Parameter(bias_tensor.detach(), requires_grad=False),
            )
        else:
            self.register_parameter("bias", None)

        if rank <= 0:
            self.register_buffer(
                "Uk",
                torch.empty(out_features, 0, device=weight.device, dtype=weight.dtype),
            )
            self.register_buffer(
                "Vk",
                torch.empty(in_features, 0, device=weight.device, dtype=weight.dtype),
            )
            self.register_parameter(
                "lora_A",
                nn.Parameter(torch.zeros(0, in_features, device=weight.device, dtype=weight.dtype)),
            )
            self.register_parameter(
                "lora_B",
                nn.Parameter(torch.zeros(out_features, 0, device=weight.device, dtype=weight.dtype)),
            )
            self._oplora_active = False
            return

        self._oplora_active = True
        kk = min(max(top_k, 0), min(out_features, in_features))
        W = self.weight.data
        if kk > 0:
            U, _s, Vh = torch.linalg.svd(W, full_matrices=False)
            kk = min(kk, U.shape[1], Vh.shape[0])
            Uk = U[:, :kk].contiguous()
            Vk = Vh[:kk, :].T.contiguous()
        else:
            Uk = torch.empty(out_features, 0, device=W.device, dtype=W.dtype)
            Vk = torch.empty(in_features, 0, device=W.device, dtype=W.dtype)

        self.register_buffer("Uk", Uk)
        self.register_buffer("Vk", Vk)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=weight.device, dtype=weight.dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=weight.device, dtype=weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        top_k: int,
        alpha: float,
    ) -> "OPLoRALinear":
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if not getattr(self, "_oplora_active", False) or self.rank <= 0:
            return out
        scale = self.alpha / self.rank
        # P_R @ x (row vectors)
        if self.Vk.shape[1] > 0:
            xv = torch.matmul(x, self.Vk)
            xr = x - torch.matmul(xv, self.Vk.t())
        else:
            xr = x
        z = F.linear(xr, self.lora_A)
        u = F.linear(z, self.lora_B)
        if self.Uk.shape[1] > 0:
            xu = torch.matmul(u, self.Uk)
            ur = u - torch.matmul(xu, self.Uk.t())
        else:
            ur = u
        return out + scale * ur


def inject_oplora_into_backbone(
    backbone: nn.Module,
    enable: bool,
    rank: int,
    top_k: int,
    alpha: float,
    target_linear_names: Optional[Sequence[str]] = None,
) -> Tuple[int, int]:
    """
    Replace selected nn.Linear modules under ``backbone`` with OPLoRALinear.

    Returns:
        (num_replaced, num_skipped_nonlinear): counts for logging.
    """
    if not enable:
        return 0, 0
    names = tuple(target_linear_names or ("qkv", "proj", "fc1", "fc2"))
    replaced = [0]

    def recurse(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in names and not isinstance(child, OPLoRALinear):
                setattr(parent, name, OPLoRALinear.from_linear(child, rank, top_k, alpha))
                replaced[0] += 1
            else:
                recurse(child)

    recurse(backbone)
    return replaced[0], 0
