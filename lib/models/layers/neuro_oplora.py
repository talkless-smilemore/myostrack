"""
NS-OPLoRA: Neuron-Selective Orthogonal Projection LoRA.

Combines OPLoRA (orthogonal-projection low-rank adaptation) with
NeuroAda's neuron-level selectivity and adds layer-wise differentiation
specifically tuned for anti-UAV tracking (small, fast targets).

Default layer config:
  - Shallow blocks 0-2  : strong adaptation (p=0.8, rank=8, all targets)
  - CE blocks 3, 6, 9   : attention-focused (p=0.6, rank=4, qkv+proj)
  - Mid blocks 4, 5      : MLP-only light adaptation (p=0.5, rank=4)
  - Deep blocks 7,8,10,11: frozen (no injection)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch.nn as nn

from lib.models.layers.oplora import NeuronSelectiveOPLoRALinear

ANTI_UAV_DEFAULT_LAYER_CONFIG: List[Dict[str, Any]] = [
    # Group A: shallow layers — critical for small-target texture/detail matching
    {
        "blocks": [0, 1, 2],
        "rank": 8,
        "top_k": 16,
        "alpha": 8.0,
        "neuron_keep_ratio": 0.8,
        "targets": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
    },
    # Group B: CE pruning layers — key decision points for fast-target localisation
    {
        "blocks": [3, 6, 9],
        "rank": 4,
        "top_k": 16,
        "alpha": 8.0,
        "neuron_keep_ratio": 0.6,
        "targets": ["attn.qkv", "attn.proj"],
    },
    # Group C: mid-level transition — part-structure refinement
    {
        "blocks": [4, 5],
        "rank": 4,
        "top_k": 16,
        "alpha": 8.0,
        "neuron_keep_ratio": 0.5,
        "targets": ["mlp.fc1", "mlp.fc2"],
    },
    # Group D: deep layers 7, 8, 10, 11 — FROZEN (no entry = no injection)
]


def inject_neuro_oplora_into_backbone(
    backbone: nn.Module,
    enable: bool,
    layer_configs: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, int]:
    """
    Inject NeuronSelectiveOPLoRALinear into the backbone following
    a per-layer configuration.

    Args:
        backbone: VisionTransformer or VisionTransformerCE instance.
        enable: master switch.
        layer_configs: list of per-group configs.  Each dict has:
            - blocks (List[int]): block indices to inject into
            - rank (int): LoRA rank
            - top_k (int): number of SVD singular vectors
            - alpha (float): scaling factor
            - neuron_keep_ratio (float): fraction of neurons to keep active
            - targets (List[str]): Linear sub-module names to replace
          If None, ANTI_UAV_DEFAULT_LAYER_CONFIG is used.

    Returns:
        (num_replaced, num_frozen): counts for logging.
    """
    if not enable:
        return 0, 0

    configs = layer_configs or ANTI_UAV_DEFAULT_LAYER_CONFIG

    replaced = 0
    num_blocks = len(backbone.blocks)

    for group_cfg in configs:
        rank = int(group_cfg["rank"])
        top_k = int(group_cfg["top_k"])
        alpha = float(group_cfg["alpha"])
        neuron_keep_ratio = float(group_cfg["neuron_keep_ratio"])
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
                        NeuronSelectiveOPLoRALinear.from_linear(
                            linear, rank, top_k, alpha, neuron_keep_ratio,
                        ),
                    )
                    replaced += 1

    frozen_blocks = _report_frozen(backbone, configs)
    return replaced, frozen_blocks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_linear(parent: nn.Module, name: str) -> Tuple[Optional[nn.Linear], Optional[nn.Module], str]:
    """Walk into nested modules (e.g. attn.qkv or mlp.fc1).

    Returns:
        (linear, parent_of_linear, leaf_attr_name) or (None, None, "")
    """
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


def _report_frozen(backbone: nn.Module, configs: List[dict]) -> int:
    """Identify and log which blocks are intentionally frozen."""
    all_injected = set()
    for g in configs:
        all_injected.update(g["blocks"])
    frozen = [i for i in range(len(backbone.blocks)) if i not in all_injected]
    if frozen:
        import logging
        _log = logging.getLogger(__name__)
        _log.info(f"NS-OPLoRA: blocks {frozen} are FROZEN (no PEFT injected)")
    return len(frozen)
