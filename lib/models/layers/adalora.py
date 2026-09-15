from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLoRALinear(nn.Module):
    """AdaLoRA wrapper for nn.Linear.

    AdaLoRA uses an SVD-style adapter:
        Delta W = B @ diag(E) @ A

    RankAllocator masks entries in E during training to adaptively allocate a
    global rank budget across all wrapped Linear layers.
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
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        if bias and bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        if self.rank <= 0:
            self.lora_A = nn.Parameter(torch.zeros(0, in_features, device=weight.device, dtype=weight.dtype))
            self.lora_E = nn.Parameter(torch.zeros(0, 1, device=weight.device, dtype=weight.dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, 0, device=weight.device, dtype=weight.dtype))
            self.register_buffer("ranknum", torch.tensor(0.0, device=weight.device, dtype=weight.dtype))
            self._active = False
            return

        self._active = True
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features, device=weight.device, dtype=weight.dtype))
        self.lora_E = nn.Parameter(torch.zeros(self.rank, 1, device=weight.device, dtype=weight.dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.rank, device=weight.device, dtype=weight.dtype))
        self.register_buffer("ranknum", torch.tensor(float(self.rank), device=weight.device, dtype=weight.dtype))

        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "AdaLoRALinear":
        max_rank = min(linear.in_features, linear.out_features)
        rank = int(rank)
        if rank <= 0 or rank > max_rank:
            raise ValueError(f"AdaLoRA rank must be in [1, {max_rank}], got {rank}.")
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
        if not getattr(self, "_active", False):
            return out
        adapter = self.dropout(x) @ (self.lora_A * self.lora_E).t() @ self.lora_B.t()
        return out + adapter * self.scaling / (self.ranknum + 1e-5)


def inject_adalora_into_backbone(
    backbone: nn.Module,
    enable: bool,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_linear_names: Optional[Sequence[str]] = None,
    freeze_backbone: bool = True,
) -> Tuple[int, int]:
    """Replace selected backbone Linear layers with AdaLoRA wrappers."""
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
            if isinstance(child, nn.Linear) and name in names and not isinstance(child, AdaLoRALinear):
                setattr(parent, name, AdaLoRALinear.from_linear(child, rank, alpha, dropout))
                replaced += 1
            else:
                recurse(child)

    recurse(backbone)

    trainable = 0
    for module in backbone.modules():
        if isinstance(module, AdaLoRALinear):
            trainable += module.lora_A.numel() + module.lora_E.numel() + module.lora_B.numel()

    return replaced, trainable


class RankAllocator:
    """AdaLoRA rank budget scheduler.

    Call update_and_mask() after backward and optimizer.step(). It uses the
    latest gradients to maintain importance estimates, then masks low-importance
    singular values in lora_E according to the scheduled rank budget.
    """

    def __init__(
        self,
        model: nn.Module,
        init_rank: int,
        target_rank: int,
        init_warmup: int,
        final_warmup: int,
        mask_interval: int,
        beta1: float,
        beta2: float,
        total_step: int,
    ):
        if not 0.0 < beta1 < 1.0:
            raise ValueError(f"beta1 must be in (0, 1), got {beta1}.")
        if not 0.0 < beta2 < 1.0:
            raise ValueError(f"beta2 must be in (0, 1), got {beta2}.")
        if total_step <= init_warmup + final_warmup:
            raise ValueError("total_step must be larger than init_warmup + final_warmup.")

        self.model = model
        self.init_rank = int(init_rank)
        self.target_rank = int(target_rank)
        self.init_warmup = int(init_warmup)
        self.final_warmup = int(final_warmup)
        self.mask_interval = max(int(mask_interval), 1)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.total_step = int(total_step)

        self.ipt: Dict[str, torch.Tensor] = {}
        self.exp_avg_ipt: Dict[str, torch.Tensor] = {}
        self.exp_avg_unc: Dict[str, torch.Tensor] = {}
        self.shape_dict: Dict[str, torch.Size] = {}
        self.rank_pattern: Dict[str, int] = {}
        self.total_rank = 0
        self._collect_lora_shapes()

    def _collect_lora_shapes(self) -> None:
        name_set = set()
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                name_set.add(name.replace("lora_A", "%s"))
                self.total_rank += param.size(0)
                self.shape_dict[name] = param.shape
            elif "lora_B" in name:
                self.shape_dict[name] = param.shape
        if not name_set:
            raise ValueError("AdaLoRA RankAllocator found no lora_A parameters.")
        self.target_total_rank = self.target_rank * len(name_set)
        self.target_total_rank = min(max(self.target_total_rank, 1), self.total_rank)

    def get_rank_pattern(self) -> Dict[str, int]:
        return dict(self.rank_pattern)

    def schedule_rank(self, step: int) -> Tuple[int, bool]:
        if step <= self.init_warmup:
            return self.total_rank, False
        if step > self.total_step - self.final_warmup:
            return self.target_total_rank, True

        denom = self.total_step - self.final_warmup - self.init_warmup
        mul_coeff = 1.0 - (step - self.init_warmup) / max(float(denom), 1.0)
        curr_rank = self.target_total_rank + (self.total_rank - self.target_total_rank) * (mul_coeff ** 3)
        return int(curr_rank), step % self.mask_interval == 0

    def update_importance(self) -> None:
        for name, param in self.model.named_parameters():
            if "lora_" not in name or param.grad is None:
                continue
            ipt = (param.detach() * param.grad.detach()).abs()
            if name not in self.ipt:
                self.ipt[name] = ipt
                self.exp_avg_ipt[name] = ipt.clone()
                self.exp_avg_unc[name] = torch.zeros_like(ipt)
            else:
                self.ipt[name] = ipt
                self.exp_avg_ipt[name] = self.beta1 * self.exp_avg_ipt[name] + (1.0 - self.beta1) * ipt
                unc = (ipt - self.exp_avg_ipt[name]).abs()
                self.exp_avg_unc[name] = self.beta2 * self.exp_avg_unc[name] + (1.0 - self.beta2) * unc

    def _score(self, name: str) -> torch.Tensor:
        if name not in self.exp_avg_ipt:
            return torch.zeros_like(dict(self.model.named_parameters())[name])
        return self.exp_avg_ipt[name] * self.exp_avg_unc[name]

    def mask_to_rank(self, curr_rank: int) -> Optional[float]:
        combined = {}
        singular = {}
        params = dict(self.model.named_parameters())

        for name, param in params.items():
            if "lora_A" in name:
                combined.setdefault(name.replace("lora_A", "%s"), []).append(self._score(name).mean(dim=1, keepdim=True))
            elif "lora_B" in name:
                combined.setdefault(name.replace("lora_B", "%s"), []).append(self._score(name).mean(dim=0).view(-1, 1))
            elif "lora_E" in name:
                singular[name.replace("lora_E", "%s")] = self._score(name).view(-1, 1)

        all_scores = []
        score_by_e = {}
        for name_mat, scores in combined.items():
            if name_mat not in singular:
                continue
            score = singular[name_mat].view(-1) + torch.cat(scores, dim=1).sum(dim=1)
            score_by_e[name_mat % "lora_E"] = score.view(-1, 1)
            all_scores.append(score.view(-1))

        if not all_scores:
            return None
        scores = torch.cat(all_scores)
        keep_total = min(max(int(curr_rank), 0), scores.numel())
        if keep_total <= 0:
            global_keep = torch.zeros_like(scores, dtype=torch.bool)
            threshold = float("inf")
        elif keep_total >= scores.numel():
            global_keep = torch.ones_like(scores, dtype=torch.bool)
            threshold = scores.min().item()
        else:
            topk = torch.topk(scores, keep_total, largest=True).indices
            global_keep = torch.zeros_like(scores, dtype=torch.bool)
            global_keep[topk] = True
            threshold = scores[topk].min().item()

        with torch.no_grad():
            offset = 0
            for name, score in score_by_e.items():
                if name not in params:
                    continue
                numel = score.numel()
                keep = global_keep[offset:offset + numel].view_as(score)
                offset += numel
                params[name].data.masked_fill_(~keep, 0.0)
                ranknum = int(keep.sum().item())
                self.rank_pattern[name] = ranknum
                module = _module_from_param_name(self.model, name)
                if isinstance(module, AdaLoRALinear):
                    module.ranknum.fill_(float(max(ranknum, 1)))

        return threshold

    def update_and_mask(self, model: Optional[nn.Module] = None, global_step: int = 0) -> Tuple[int, Optional[float]]:
        if model is not None and model is not self.model:
            self.model = model
        if global_step < self.total_step - self.final_warmup:
            self.update_importance()
        curr_rank, should_mask = self.schedule_rank(global_step)
        threshold = self.mask_to_rank(curr_rank) if should_mask else None
        return curr_rank, threshold


def _module_from_param_name(model: nn.Module, param_name: str) -> Optional[nn.Module]:
    parts = param_name.split(".")[:-1]
    module: nn.Module = model
    for part in parts:
        if part.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
            module = module[int(part)]
        elif hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module


def compute_adalora_orth_regu(model: nn.Module) -> torch.Tensor:
    regu_loss = None
    num_param = 0
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if param.numel() == 0:
            continue
        cov = param @ param.t() if "lora_A" in name else param.t() @ param
        eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        loss = torch.norm(cov - eye, p="fro")
        regu_loss = loss if regu_loss is None else regu_loss + loss
        num_param += 1
    if regu_loss is None:
        ref = next(model.parameters())
        return torch.tensor(0.0, device=ref.device)
    return regu_loss / max(num_param, 1)
