# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.utils import to_2tuple

from mmdet.registry import MODELS
from ..layers import PatchEmbed, PatchMerging
from .swin import SwinTransformer

@MODELS.register_module()
class SwinTransformerForSimMIM(SwinTransformer):
    """
    扩展 mmdet 的 SwinTransformer 用于 SimMIM 自监督预训练任务，
    在 patch embedding 后根据输入 mask 替换被遮挡的 patch 特征。
    """
    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None,**kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        # 确保模型不包含分类头
        self.num_classes = 0
        # 定义一个可学习的 mask_token，尺寸为 (1, 1, embed_dims)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        trunc_normal_init(self.mask_token, std=0.02)

    def forward(self, x,mask):
        """
        Args:
            x (Tensor): 输入图像，形状 (B, 3, H, W)
            mask,输入的掩码
        Returns:
            x: 编码器输出的 patch 特征，形状 (B, L, C)
            hw_shape: 空间尺寸 (H_enc, W_enc)
        """
        # 1. Patch embedding，返回 (x, hw_shape)
        # mask (Tensor, optional): 掩码，形状 (B, num_patches)，
        #                              每个元素为 1 表示该 patch 需要被 mask 掉
        x, hw_shape = self.patch_embed(x)  # x: (B, L, embed_dims)

        # 2. 处理掩码
        B, L, C = x.shape

        mask = mask.view(B, -1)  # 将掩码形状从 (B, H_enc, W_enc) 展平为 (B, L)
    
        # 应用掩码：掩码为 0 的位置替换为 mask_token
        mask_tokens = self.mask_token.expand(B, L, -1).to(x.device)  # 扩展 mask_token 到 (B, L, C)
    
        for b in range(B):
            # 对应的掩码位置标记为0，替换为 mask_token
            x[b, mask[b] == 0] = mask_tokens[b, mask[b] == 0]

        # 3. 后续步骤按原 SwinTransformer 流程执行
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
        last_out = outs[-1]
        return last_out
