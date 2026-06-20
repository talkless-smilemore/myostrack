from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Vanilla LoRA wrapper for nn.Linear.

    Forward:
        y = x W0^T + b + (alpha / rank) * (x A^T) B^T

    The original linear weight and bias are frozen. Only lora_A and lora_B
    are trainable, matching the original LoRA baseline.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        rank: int,
        alpha: float,
        dropout: float,
        weight: torch.Tensor,
        bias_tensor: Optional[torch.Tensor],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank if self.rank > 0 else 0.0
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        if bias and bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        if self.rank <= 0:
            self.lora_A = nn.Parameter(torch.zeros(0, in_features, device=weight.device, dtype=weight.dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, 0, device=weight.device, dtype=weight.dtype))
            self._lora_active = False
            return

        self._lora_active = True
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features, device=weight.device, dtype=weight.dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.rank, device=weight.device, dtype=weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        has_bias = linear.bias is not None
        return cls(
            linear.in_features,
            linear.out_features,
            has_bias,
            rank,
            alpha,
            dropout,
            linear.weight.data,
            linear.bias.data if has_bias else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if not getattr(self, "_lora_active", False):
            return out
        z = F.linear(self.dropout(x), self.lora_A)
        u = F.linear(z, self.lora_B)
        return out + self.scaling * u


def inject_lora_into_backbone(
    backbone: nn.Module,
    enable: bool,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_linear_names: Optional[Sequence[str]] = None,
    freeze_backbone: bool = True,
) -> Tuple[int, int]:
    """Replace selected backbone Linear layers with vanilla LoRA wrappers.

    Args:
        backbone: ViT backbone.
        enable: global switch.
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: dropout before LoRA A.
        target_linear_names: leaf names such as qkv, proj, fc1, fc2.
        freeze_backbone: freeze all original backbone parameters except LoRA A/B.

    Returns:
        (num_replaced, num_trainable_lora_params)
    """
    if not enable:
        return 0, 0

    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)

    names = tuple(target_linear_names or ("qkv", "proj", "fc1", "fc2"))
    replaced = 0

    def recurse(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in names and not isinstance(child, LoRALinear):
                setattr(parent, name, LoRALinear.from_linear(child, rank, alpha, dropout))
                replaced += 1
            else:
                recurse(child)

    recurse(backbone)

    trainable = 0
    for module in backbone.modules():
        if isinstance(module, LoRALinear):
            trainable += module.lora_A.numel() + module.lora_B.numel()

    return replaced, trainable
