import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14,
                 evt_enable=False, evt_gamma=0.875, evt_apply_cross=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe =rpe
        self.evt_enable = evt_enable
        self.evt_apply_cross = evt_apply_cross
        gamma = torch.as_tensor(evt_gamma, dtype=torch.float32).flatten()
        if gamma.numel() == 1:
            gamma = gamma.repeat(num_heads)
        elif gamma.numel() != num_heads:
            raise ValueError(f"EVT gamma must be a scalar or have {num_heads} values, got {gamma.numel()}.")
        self.register_buffer("evt_gamma", gamma.clamp(1e-6, 1.0))
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _coords_from_index(self, index, grid_size):
        h, w = grid_size
        row = torch.div(index, w, rounding_mode='floor')
        col = index - row * w
        return torch.stack((row, col), dim=-1).float()

    def _evt_log_decay(self, template_index, search_index, template_grid_size, search_grid_size, dtype):
        coords_t = self._coords_from_index(template_index, template_grid_size)
        coords_s = self._coords_from_index(search_index, search_grid_size)
        coords = torch.cat((coords_t, coords_s), dim=1)
        dist = torch.cdist(coords, coords, p=2)

        lens_t = template_index.shape[1]
        lens_s = search_index.shape[1]
        group_mask = torch.zeros((coords.shape[0], lens_t + lens_s, lens_t + lens_s),
                                 device=coords.device, dtype=torch.bool)
        group_mask[:, :lens_t, :lens_t] = True
        group_mask[:, lens_t:, lens_t:] = True
        if self.evt_apply_cross:
            group_mask[:, :lens_t, lens_t:] = True
            group_mask[:, lens_t:, :lens_t] = True

        gamma = self.evt_gamma.to(device=coords.device, dtype=dtype).view(1, self.num_heads, 1, 1)
        log_decay = torch.log(gamma) * dist.to(dtype).unsqueeze(1)
        return log_decay.masked_fill(~group_mask.unsqueeze(1), 0.0)

    def forward(self, x, mask=None, return_attention=False,
                evt_template_index=None, evt_search_index=None,
                evt_template_grid_size=None, evt_search_grid_size=None):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if self.evt_enable:
            if (evt_template_index is None or evt_search_index is None or
                    evt_template_grid_size is None or evt_search_grid_size is None):
                raise ValueError("EVT attention requires template/search indices and grid sizes.")
            attn += self._evt_log_decay(evt_template_index, evt_search_index,
                                        evt_template_grid_size, evt_search_grid_size, attn.dtype)

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
