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
from lib.models.layers.oplora import inject_oplora_into_backbone
from lib.models.layers.neuro_oplora import inject_neuro_oplora_into_backbone, ANTI_UAV_DEFAULT_LAYER_CONFIG
from lib.models.layers.sglora import inject_sglora_into_backbone, SGLORA_DEFAULT_LAYER_CONFIG
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

    # NS-OPLoRA (neuron-selective + layer-selective): takes priority over plain OPLoRA
    neuro_cfg = getattr(cfg.TRAIN, "NEURO_OPLORA", None)
    oplora_cfg = getattr(cfg.TRAIN, "OPLORA", None)
    if neuro_cfg is not None and getattr(neuro_cfg, "ENABLE", False):
        layer_configs = getattr(neuro_cfg, "LAYER_CONFIGS", None) or None
        if layer_configs is None:
            layer_configs = ANTI_UAV_DEFAULT_LAYER_CONFIG
        n_rep, n_frozen = inject_neuro_oplora_into_backbone(
            model.backbone,
            enable=True,
            layer_configs=layer_configs,
        )
        print(f"NS-OPLoRA: replaced {n_rep} Linear layers, {n_frozen} blocks frozen.")

    elif oplora_cfg is not None and getattr(oplora_cfg, "ENABLE", False):
        n_rep, _ = inject_oplora_into_backbone(
            model.backbone,
            enable=True,
            rank=int(oplora_cfg.RANK),
            top_k=int(oplora_cfg.TOP_K),
            alpha=float(oplora_cfg.ALPHA),
            target_linear_names=getattr(oplora_cfg, "TARGETS", None),
        )
        print(f"OPLoRA: replaced {n_rep} Linear layers in backbone (rank={oplora_cfg.RANK}, top_k={oplora_cfg.TOP_K}).")

    # SGLoRA (Spectral-Gated LoRA): deeply fused PEFT — replaces OPLoRA+NeuroAda stack
    sglora_cfg = getattr(cfg.TRAIN, "SGLORA", None)
    if sglora_cfg is not None and getattr(sglora_cfg, "ENABLE", False):
        layer_configs = getattr(sglora_cfg, "LAYER_CONFIGS", None) or None
        if layer_configs is None:
            layer_configs = SGLORA_DEFAULT_LAYER_CONFIG
        n_rep, n_frozen = inject_sglora_into_backbone(
            model.backbone,
            enable=True,
            layer_configs=layer_configs,
            entropy_lam_max=float(getattr(sglora_cfg, "ENTROPY_LAM_MAX", 1e-4)),
            warmup_ratio=float(getattr(sglora_cfg, "WARMUP_RATIO", 0.33)),
            anneal_ratio=float(getattr(sglora_cfg, "ANNEAL_RATIO", 0.33)),
            group_lasso_lam_max=float(getattr(sglora_cfg, "GROUP_LASSO_LAM_MAX", 1e-5)),
        )
        print(f"SGLoRA: replaced {n_rep} Linear layers, {n_frozen} blocks frozen "
              f"(entropy_lam_max={getattr(sglora_cfg, 'ENTROPY_LAM_MAX', 1e-4)}, "
              f"warmup={getattr(sglora_cfg, 'WARMUP_RATIO', 0.33)}, "
              f"anneal={getattr(sglora_cfg, 'ANNEAL_RATIO', 0.33)}).")

    return model
