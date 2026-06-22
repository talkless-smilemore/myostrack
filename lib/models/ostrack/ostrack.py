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
from lib.models.layers.lora import inject_lora_into_backbone
from lib.models.layers.lora_null import inject_lora_null_into_backbone
from lib.models.layers.milora import inject_milora_into_backbone
from lib.models.layers.uav_wsp import inject_wsp_into_backbone, WSP_DEFAULT_PRIOR_CONFIG
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """
#鍒濆鍖栦竴涓?OSTrack 妯″瀷锛屾妸 backbone 鍜?box head 瀛樿捣鏉ャ€?
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
# 瀹屾暣鍓嶅悜浼犳挱銆傝緭鍏ユā鏉垮浘鍜屾悳绱㈠浘锛岃緭鍑洪娴嬫銆乻core map 绛夌粨鏋溿€?
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
# 浠?backbone 杈撳嚭閲屽彇鍑烘悳绱㈠尯鍩?token锛屽苟閫佸叆棰勬祴澶村緱鍒版銆?
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
    wsp_cfg = getattr(cfg.TRAIN, "UAV_WSP", None)
    active_adapters = [
        name for name, enabled in (
            ("LORA", lora_cfg is not None and getattr(lora_cfg, "ENABLE", False)),
            ("LORA_NULL", lora_null_cfg is not None and getattr(lora_null_cfg, "ENABLE", False)),
            ("MILORA", milora_cfg is not None and getattr(milora_cfg, "ENABLE", False)),
            ("UAV_WSP", wsp_cfg is not None and getattr(wsp_cfg, "ENABLE", False)),
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
    # WSP (Weighted Spectral Projection): anti-UAV small-target adapter
    # Unified PEFT with spectrally weighted orthogonal projection,
    # target-saliency gates, and focus regularisation.
    if wsp_cfg is not None and getattr(wsp_cfg, "ENABLE", False):
        layer_configs = getattr(wsp_cfg, "LAYER_CONFIGS", None) or None
        if layer_configs is None:
            layer_configs = WSP_DEFAULT_PRIOR_CONFIG
        n_rep, n_frozen = inject_wsp_into_backbone(
            model.backbone,
            enable=True,
            layer_configs=layer_configs,
            entropy_lam_max=float(getattr(wsp_cfg, "ENTROPY_LAM_MAX", 1e-4)),
            warmup_ratio=float(getattr(wsp_cfg, "WARMUP_RATIO", 0.33)),
            anneal_ratio=float(getattr(wsp_cfg, "ANNEAL_RATIO", 0.33)),
            group_lasso_lam_max=float(getattr(wsp_cfg, "GROUP_LASSO_LAM_MAX", 1e-5)),
            focus_lam_max=float(getattr(wsp_cfg, "FOCUS_LAM_MAX", 1e-5)),
        )
        beta_info = getattr(wsp_cfg, "SPECTRAL_BETA", "per-group")
        print(f"UAV-WSP: replaced {n_rep} Linear layers, {n_frozen} blocks frozen "
              f"(spectral_beta={beta_info}, "
              f"entropy_lam_max={getattr(wsp_cfg, 'ENTROPY_LAM_MAX', 1e-4)}, "
              f"focus_lam_max={getattr(wsp_cfg, 'FOCUS_LAM_MAX', 1e-5)}).")

    return model


