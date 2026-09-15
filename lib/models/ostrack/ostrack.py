"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.layers.asc_lora import inject_asc_lora_into_backbone, ASC_LORA_DEFAULT_PRIOR_CONFIG
from lib.models.layers.lora import inject_lora_into_backbone
from lib.models.layers.lora_null import inject_lora_null_into_backbone
from lib.models.layers.milora import inject_milora_into_backbone
from lib.models.layers.adalora import inject_adalora_into_backbone
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """
#初始化一个 OSTrack 模型，把 backbone 和 box head 存起来。
    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
# 完整前向传播。输入模板图和搜索图，输出预测框、score map 等结果。
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out
# 从 backbone 输出里取出搜索区域 token，并送入预测头得到框。
    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu", weights_only=False)
        ckpt_state = checkpoint["net"]
        model_state = model.state_dict()
        # Allow loading checkpoints trained with different template/search sizes
        # by skipping parameters whose tensor shapes do not match.
        incompatible_keys = []
        for k in list(ckpt_state.keys()):
            if k in model_state and ckpt_state[k].shape != model_state[k].shape:
                incompatible_keys.append(k)
                ckpt_state.pop(k)
        if incompatible_keys:
            print("Skip incompatible pretrained keys:", incompatible_keys)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt_state, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    lora_cfg = getattr(cfg.TRAIN, "LORA", None)
    lora_null_cfg = getattr(cfg.TRAIN, "LORA_NULL", None)
    milora_cfg = getattr(cfg.TRAIN, "MILORA", None)
    adalora_cfg = getattr(cfg.TRAIN, "ADALORA", None)
    asc_lora_cfg = getattr(cfg.TRAIN, "ASC_LORA", None)
    active_adapters = [
        name for name, enabled in (
            ("LORA", lora_cfg is not None and getattr(lora_cfg, "ENABLE", False)),
            ("LORA_NULL", lora_null_cfg is not None and getattr(lora_null_cfg, "ENABLE", False)),
            ("MILORA", milora_cfg is not None and getattr(milora_cfg, "ENABLE", False)),
            ("ADALORA", adalora_cfg is not None and getattr(adalora_cfg, "ENABLE", False)),
            ("ASC_LORA", asc_lora_cfg is not None and getattr(asc_lora_cfg, "ENABLE", False)),
        )
        if enabled
    ]
    if len(active_adapters) > 1:
        raise ValueError(f"Enable only one adapter baseline at a time, got: {active_adapters}")
    
    # Vanilla LoRA baseline: freeze original backbone weights and train only A/B adapters.
    if lora_cfg is not None and getattr(lora_cfg, "ENABLE", False):
        n_rep, n_lora_params = inject_lora_into_backbone(
            model.backbone,
            enable=True,
            rank=int(getattr(lora_cfg, "RANK", 8)),
            alpha=float(getattr(lora_cfg, "ALPHA", 8.0)),
            dropout=float(getattr(lora_cfg, "DROPOUT", 0.0)),
            target_linear_names=getattr(lora_cfg, "TARGETS", ["qkv", "proj", "fc1", "fc2"]),
            freeze_backbone=bool(getattr(lora_cfg, "FREEZE_BACKBONE", True)),
        )
        print(f"Vanilla LoRA: replaced {n_rep} Linear layers, trainable LoRA params={n_lora_params}.")

    # LoRA-Null baseline: SVD residual + trainable low-rank factors.
    if lora_null_cfg is not None and getattr(lora_null_cfg, "ENABLE", False):
        n_rep, n_lora_null_params = inject_lora_null_into_backbone(
            model.backbone,
            enable=True,
            rank=int(getattr(lora_null_cfg, "RANK", 8)),
            alpha=float(getattr(lora_null_cfg, "ALPHA", 1.0)),
            target_linear_names=getattr(lora_null_cfg, "TARGETS", ["qkv", "proj", "fc1", "fc2"]),
            freeze_backbone=bool(getattr(lora_null_cfg, "FREEZE_BACKBONE", True)),
            use_last=bool(getattr(lora_null_cfg, "USE_LAST", True)),
        )
        print(f"LoRA-Null: replaced {n_rep} Linear layers, trainable params={n_lora_null_params}.")

# MiLoRA baseline: freeze principal singular components and train minor components.
    if milora_cfg is not None and getattr(milora_cfg, "ENABLE", False):
        n_rep, n_milora_params = inject_milora_into_backbone(
            model.backbone,
            enable=True,
            rank=int(getattr(milora_cfg, "RANK", 8)),
            alpha=float(getattr(milora_cfg, "ALPHA", 1.0)),
            target_linear_names=getattr(milora_cfg, "TARGETS", ["qkv", "proj", "fc1", "fc2"]),
            freeze_backbone=bool(getattr(milora_cfg, "FREEZE_BACKBONE", True)),
        )
        print(f"MiLoRA: replaced {n_rep} Linear layers, trainable params={n_milora_params}.")
    # AdaLoRA baseline: SVD-style LoRA with adaptive rank allocation during training.
    if adalora_cfg is not None and getattr(adalora_cfg, "ENABLE", False):
        n_rep, n_adalora_params = inject_adalora_into_backbone(
            model.backbone,
            enable=True,
            rank=int(getattr(adalora_cfg, "INIT_RANK", 12)),
            alpha=float(getattr(adalora_cfg, "ALPHA", 8.0)),
            dropout=float(getattr(adalora_cfg, "DROPOUT", 0.0)),
            target_linear_names=getattr(adalora_cfg, "TARGETS", ["qkv", "proj", "fc1", "fc2"]),
            freeze_backbone=bool(getattr(adalora_cfg, "FREEZE_BACKBONE", True)),
        )
        print(f"AdaLoRA: replaced {n_rep} Linear layers, trainable params={n_adalora_params}.")
    # ASC-LoRA: anti-UAV small-target adapter
    # Unified PEFT with spectrally weighted orthogonal projection,
    # target-saliency gates, and focus regularisation.

    if asc_lora_cfg is not None and getattr(asc_lora_cfg, "ENABLE", False):
        # ASC-LoRA ablations freeze the original backbone and train only the
        # injected adapter parameters plus the tracking head. ASCLoRA's
        # lora_A/lora_B/gate parameters are created trainable, while its
        # copied base weight/bias stay frozen.
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

        layer_configs = getattr(asc_lora_cfg, "LAYER_CONFIGS", None) or None
        if layer_configs is None:
            layer_configs = ASC_LORA_DEFAULT_PRIOR_CONFIG
        rank_override = getattr(asc_lora_cfg, "RANK", None)
        if rank_override is not None:
            rank_override = int(rank_override)
            # Keep the current best SACT layer selection and all per-group
            # settings fixed; only replace the rank for this experiment.
            layer_configs = [dict(group_cfg, rank=rank_override)
                             for group_cfg in layer_configs]
        n_rep, n_unadapted = inject_asc_lora_into_backbone(
            model.backbone,
            enable=True,
            layer_configs=layer_configs,
            entropy_lam_max=float(getattr(asc_lora_cfg, "ENTROPY_LAM_MAX", 1e-4)),
            warmup_ratio=float(getattr(asc_lora_cfg, "WARMUP_RATIO", 0.33)),
            anneal_ratio=float(getattr(asc_lora_cfg, "ANNEAL_RATIO", 0.33)),
            group_lasso_lam_max=float(getattr(asc_lora_cfg, "GROUP_LASSO_LAM_MAX", 1e-5)),
            focus_lam_max=float(getattr(asc_lora_cfg, "FOCUS_LAM_MAX", 1e-5)),
            channel_gate=bool(getattr(asc_lora_cfg, "CHANNEL_GATE", True)),
            spectral_projection=str(getattr(asc_lora_cfg, "SPECTRAL_PROJECTION", "weighted")),
            dropout=float(getattr(asc_lora_cfg, "DROPOUT", 0.0)),
        )
        beta_info = [g.get("spectral_beta", 1.0) for g in layer_configs]
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("ASC-LoRA experiment: "
              f"name={getattr(asc_lora_cfg, 'EXPERIMENT_NAME', 'ASC-LoRA')}; "
              f"blocks={[g['blocks'] for g in layer_configs]}; "
              f"ranks={[g['rank'] for g in layer_configs]}; "
              f"channel_gate={getattr(asc_lora_cfg, 'CHANNEL_GATE', True)}; "
              f"spectral_projection={getattr(asc_lora_cfg, 'SPECTRAL_PROJECTION', 'weighted')}; "
              f"top_k={[g['top_k'] for g in layer_configs]}; beta={beta_info}; "
              f"complement_loss_weight={getattr(asc_lora_cfg, 'COMPLEMENT_LOSS_WEIGHT', 1e-4)}; "
              f"trainable_parameters={trainable}.")
        print(f"ASC-LoRA: replaced {n_rep} Linear layers; {n_unadapted} blocks use standard fine-tuning.")

    return model
