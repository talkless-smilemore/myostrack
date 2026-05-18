"""
NeuroAda for parameter-efficient fine-tuning.

For each output neuron of a frozen Linear layer, NeuroAda keeps a trainable
bypass only on the top-k input weights with the largest magnitudes. The bypass
is zero-initialized, so the wrapped layer starts as the original layer.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sparse_neuroada_bypass(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    delta: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    orig_shape = x.shape[:-1]
    x_flat = x.reshape(-1, x.shape[-1])
    selected = x_flat.index_select(1, topk_indices.reshape(-1))
    selected = selected.view(x_flat.shape[0], out_features, topk_indices.shape[1])
    bypass = (selected * delta.unsqueeze(0)).sum(dim=-1)
    return bypass.view(*orig_shape, out_features)


class NeuroAdaLinear(nn.Module):
    """Frozen Linear plus sparse trainable per-neuron bypass."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        top_k: int,
        scale: float,
        weight: torch.Tensor,
        bias_tensor: Optional[torch.Tensor],
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.top_k = int(top_k)
        self.scale = float(scale)

        self.register_parameter("weight", nn.Parameter(weight.detach(), requires_grad=False))
        if bias and bias_tensor is not None:
            self.register_parameter("bias", nn.Parameter(bias_tensor.detach(), requires_grad=False))
        else:
            self.register_parameter("bias", None)

        kk = min(max(self.top_k, 0), self.in_features)
        if kk <= 0:
            self.register_buffer(
                "topk_indices",
                torch.empty(self.out_features, 0, device=weight.device, dtype=torch.long),
            )
            self.delta = nn.Parameter(torch.zeros(self.out_features, 0, device=weight.device, dtype=weight.dtype))
            self._neuroada_active = False
            return

        topk_indices = torch.topk(weight.detach().abs(), k=kk, dim=1, largest=True).indices.contiguous()
        self.register_buffer("topk_indices", topk_indices)
        self.delta = nn.Parameter(torch.zeros(self.out_features, kk, device=weight.device, dtype=weight.dtype))
        self._neuroada_active = True

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        top_k: int,
        scale: float,
    ) -> "NeuroAdaLinear":
        has_bias = linear.bias is not None
        return cls(
            linear.in_features,
            linear.out_features,
            has_bias,
            top_k,
            scale,
            linear.weight.data,
            linear.bias.data if has_bias else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if not getattr(self, "_neuroada_active", False) or self.delta.numel() == 0:
            return out

        bypass = _sparse_neuroada_bypass(x, self.topk_indices, self.delta, self.out_features)
        return out + self.scale * bypass


class NeuroAdaWrappedLinear(nn.Module):
    """Add NeuroAda sparse bypass on top of a linear-like base module."""

    def __init__(self, base_layer: nn.Module, top_k: int, scale: float):
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise TypeError("NeuroAdaWrappedLinear requires a base layer with a weight parameter.")

        self.base_layer = base_layer
        weight = base_layer.weight.detach()
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.top_k = int(top_k)
        self.scale = float(scale)

        kk = min(max(self.top_k, 0), self.in_features)
        if kk <= 0:
            self.register_buffer(
                "topk_indices",
                torch.empty(self.out_features, 0, device=weight.device, dtype=torch.long),
            )
            self.delta = nn.Parameter(torch.zeros(self.out_features, 0, device=weight.device, dtype=weight.dtype))
            self._neuroada_active = False
            return

        topk_indices = torch.topk(weight.abs(), k=kk, dim=1, largest=True).indices.contiguous()
        self.register_buffer("topk_indices", topk_indices)
        self.delta = nn.Parameter(torch.zeros(self.out_features, kk, device=weight.device, dtype=weight.dtype))
        self._neuroada_active = True

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.base_layer, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        if not getattr(self, "_neuroada_active", False) or self.delta.numel() == 0:
            return out

        bypass = _sparse_neuroada_bypass(x, self.topk_indices, self.delta, self.out_features)
        return out + self.scale * bypass


def inject_neuroada_into_backbone(
    backbone: nn.Module,
    enable: bool,
    top_k: int,
    scale: float = 1.0,
    target_linear_names: Optional[Sequence[str]] = None,
) -> Tuple[int, int]:
    """
    Replace selected nn.Linear modules under ``backbone`` with NeuroAdaLinear.

    Returns:
        (num_replaced, num_skipped_wrapped): counts for logging.
    """
    if not enable:
        return 0, 0
    names = tuple(target_linear_names or ("qkv", "proj", "fc1", "fc2"))
    replaced = [0]
    skipped = [0]

    def recurse(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, NeuroAdaWrappedLinear) or isinstance(child, NeuroAdaLinear):
                skipped[0] += 1
            elif name in names and hasattr(child, "weight") and child.weight.dim() == 2:
                if isinstance(child, nn.Linear):
                    setattr(parent, name, NeuroAdaLinear.from_linear(child, top_k, scale))
                else:
                    setattr(parent, name, NeuroAdaWrappedLinear(child, top_k, scale))
                replaced[0] += 1
            else:
                recurse(child)

    recurse(backbone)
    return replaced[0], skipped[0]
