from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRANullLinear(nn.Module):
    """SVD-initialized LoRA-Null wrapper for nn.Linear."""

    def __init__(
        self,
        weight_residual: torch.Tensor,
        bias_tensor: Optional[torch.Tensor],
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.in_features = int(weight_residual.size(1))
        self.out_features = int(weight_residual.size(0))
        self.rank = int(lora_A.size(0))
        self.alpha = float(alpha)

        self.weight_residual = nn.Parameter(weight_residual.detach().clone(), requires_grad=False)
        if bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(lora_A.detach().clone())
        self.lora_B = nn.Parameter(lora_B.detach().clone())

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        use_last: bool = True,
    ) -> "LoRANullLinear":
        rank = int(rank)
        max_rank = min(linear.in_features, linear.out_features)
        if rank <= 0 or rank > max_rank:
            raise ValueError(f"LoRA-Null rank must be in [1, {max_rank}], got {rank}.")

        dtype = linear.weight.dtype
        device = linear.weight.device
        weight_f = linear.weight.detach().float()

        u, s, vh = torch.linalg.svd(weight_f, full_matrices=False)
        if use_last:
            u_r = u[:, -rank:]
            s_r = s[-rank:]
            vh_r = vh[-rank:, :]
        else:
            u_r = u[:, :rank]
            s_r = s[:rank]
            vh_r = vh[:rank, :]

        sqrt_s = torch.sqrt(torch.clamp(s_r, min=0.0))
        lora_A = vh_r * sqrt_s.view(-1, 1)
        lora_B = u_r * sqrt_s.view(1, -1)
        residual = weight_f - float(alpha) * (lora_B @ lora_A)

        bias_tensor = linear.bias.detach() if linear.bias is not None else None
        return cls(
            residual.to(device=device, dtype=dtype),
            bias_tensor.to(device=device, dtype=dtype) if bias_tensor is not None else None,
            lora_A.to(device=device, dtype=dtype),
            lora_B.to(device=device, dtype=dtype),
            alpha=alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight_residual, self.bias)
        adapter = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out + self.alpha * adapter


def inject_lora_null_into_backbone(
    backbone: nn.Module,
    enable: bool,
    rank: int,
    alpha: float = 1.0,
    target_linear_names: Optional[Sequence[str]] = None,
    freeze_backbone: bool = True,
    use_last: bool = True,
) -> Tuple[int, int]:
    """Replace selected backbone Linear layers with LoRA-Null wrappers."""
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
            if isinstance(child, nn.Linear) and name in names and not isinstance(child, LoRANullLinear):
                setattr(parent, name, LoRANullLinear.from_linear(child, rank, alpha, use_last))
                replaced += 1
            else:
                recurse(child)

    recurse(backbone)

    trainable = 0
    for module in backbone.modules():
        if isinstance(module, LoRANullLinear):
            trainable += module.lora_A.numel() + module.lora_B.numel()

    return replaced, trainable
