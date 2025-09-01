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
from .lora_layers import MergedLinear
import math
from functools import reduce
from operator import mul

class AdapterFFN(FFN):
    def __init__(self, *args, **kwargs):
        # 调用父类MMCV_FFN的__init__方法
        # *args 和 **kwargs 会透传所有原始FFN的参数
        super().__init__(*args, **kwargs)
        
        # 存储传入的adapter模块实例
        self.adapter = Adapter(self.embed_dims)

    def forward(self, x, identity=None):
        out = self.layers(x)
        out = self.gamma2(out)
        # if self.adapter is not None:
        #     # 'out' 此刻是FFN核心MLP变换的输出，正是adapter_module2期望作用的对象
        out = self.adapter(out) ## Adapte的forward
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
        lora_mode (bool, optional): Whether to use lora. Default: False.
        num_prompts (int, optional): Number of prompts. Default: None.
        prompt_location (str, optional): Prompt location. Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None,
                 lora_mode=False,
                 num_prompts=None,
                 prompt_location=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg
        #vpt参数
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)
        if lora_mode:
            self.qkv = MergedLinear(embed_dims, embed_dims * 3, r=64, enable_lora=[True, False, True],
                                    bias=qkv_bias)
        else:
            self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # VPT兼容：如果有prompt参数，扩展relative_position_bias
        if self.num_prompts is not None and self.prompt_location == 'prepend':
            _C, _H, _W = relative_position_bias.shape
            relative_position_bias = torch.cat((torch.zeros(_C, self.num_prompts, _W, device=attn.device), relative_position_bias), dim=1)
            relative_position_bias = torch.cat((torch.zeros(_C, _H + self.num_prompts, self.num_prompts, device=attn.device), relative_position_bias), dim=-1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            if self.num_prompts is not None and self.prompt_location == 'prepend':
                mask = torch.cat((torch.zeros(nW, self.num_prompts, _W, device=attn.device), mask), dim=1)
                mask = torch.cat((torch.zeros(nW, _H + self.num_prompts, self.num_prompts, device=attn.device), mask), dim=-1)
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
        lora_mode (bool, optional): Whether to use lora. Default: False.
        adapter_mode (bool, optional): Whether to use adapter. Default: False.
        num_prompts (int, optional): Number of prompts. Default: None.
        prompt_location (str, optional): Prompt location. Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None,
                 lora_mode=False,
                 adapter_mode=False,
                 num_prompts=None,
                 prompt_location=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size
        self.adapter_mode = adapter_mode
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if self.adapter_mode:
            self.adapter = Adapter(embed_dims)
        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None,
            lora_mode=lora_mode,
            num_prompts=num_prompts,
            prompt_location=prompt_location)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        # VPT兼容：如果有prompt参数，分离prompt token
        if self.num_prompts is not None and self.prompt_location == 'prepend':
            prompt_emb = query[:, :self.num_prompts, :]
            query = query[:, self.num_prompts:, :]
            L = L - self.num_prompts
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # VPT兼容：如果有prompt参数，拼接prompt token
        if self.num_prompts is not None and self.prompt_location == 'prepend':
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(int(query_windows.shape[0] / B), -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))
            query_windows = torch.cat((prompt_emb, query_windows), dim=1)
        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # seperate prompt embs --> nW*B, num_prompts, C
        if self.num_prompts is not None and self.prompt_location == 'prepend':
            # change input size
            prompt_emb = attn_windows[:, :self.num_prompts, :]
            attn_windows = attn_windows[:, self.num_prompts:, :]
            # change prompt_embs's shape:
            # nW*B, num_prompts, C -> B, num_prompts, C
            prompt_emb = prompt_emb.view(-1, B, self.num_prompts, C)
            prompt_emb = prompt_emb.mean(0)


        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        # VPT兼容：如果有prompt参数，拼接输出prompt token
        if self.num_prompts is not None and self.prompt_location == 'prepend':
            x = torch.cat((prompt_emb, x), dim=1)
        if self.adapter_mode:
            x = self.adapter(x)
        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
        requires_grad (bool): Whether to require gradients.
            Default: False.
        finetune_mode (str): The mode of finetuning.
            Default: None.
        num_prompts (int, optional): Number of prompts. Default: None.
        prompt_location (str, optional): Prompt location. Default: None.
        text_dim (int, optional): Text dimension. Default: 256.
        text_consistency_loss (bool): Whether to use text consistency loss. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 requires_grad=False,
                 finetune_mode=None,
                 num_prompts=None,
                 prompt_location=None,
                 text_dim=256,
                 text_consistency_loss=False,
                 tokens_of_interest=None):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp
        self.text_consistency_loss = text_consistency_loss
        lora_mode = False
        if finetune_mode == 'lora':
            lora_mode = True
        adapter_mode = False
        if finetune_mode == 'adapter':
            adapter_mode = True
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None,
            lora_mode=lora_mode,
            adapter_mode=adapter_mode,
            num_prompts=num_prompts,
            prompt_location=prompt_location)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        if adapter_mode:
            self.ffn = AdapterFFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=2,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                add_identity=True,
                init_cfg=None)
        else:
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=2,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                add_identity=True,
                init_cfg=None)
        self.requires_grad = requires_grad
        self.finetune_mode = finetune_mode 
        # if not self.requires_grad:
        #     for param in self.parameters():
        #         param.requires_grad = False
        if self.finetune_mode is None:
            for name, param in self.named_parameters():
                param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'mona':
            self.mona_module1 = Mona(embed_dims, 8)
            self.mona_module2 = Mona(embed_dims, 8) # Adapter_FFN(dim, 8)
            for name, param in self.named_parameters():
                if 'mona_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'odmona':
            self.odmona_module1 = ODMona(embed_dims, 8)
            self.odmona_module2 = ODMona(embed_dims, 8)
            for name, param in self.named_parameters():
                if 'odmona_module' not in name:
                    param.requires_grad = self.requires_grad   
        elif self.finetune_mode == 'seeker':
            self.seeker_module1 = DynamicSeekerAdapter(embed_dims,text_dim,text_consistency_loss=text_consistency_loss,tokens_of_interest=tokens_of_interest)
            self.seeker_module2 = DynamicSeekerAdapter(embed_dims,text_dim,text_consistency_loss=text_consistency_loss,tokens_of_interest=tokens_of_interest)
            for name, param in self.named_parameters():
                if 'seeker_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'visual_seeker':
            self.visual_seeker_module1 = VisualSeekerAdapter(embed_dims)
            self.visual_seeker_module2 = VisualSeekerAdapter(embed_dims)
            for name, param in self.named_parameters():
                if 'visual_seeker_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'omni_seeker':
            self.omni_seeker_module1 = OmniDynamicSeekerAdapter(embed_dims,text_dim)
            self.omni_seeker_module2 = OmniDynamicSeekerAdapter(embed_dims,text_dim)
            for name, param in self.named_parameters():
                if 'omni_seeker_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'adapter':
            for name, param in self.named_parameters():
                if 'adapter' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'adapter_former':
            self.adapter_former_module1 = AdapterFormer(embed_dims)
            for name, param in self.named_parameters():
                if 'adapter_former_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'bitfit':
            for name, param in self.named_parameters():
                if 'bias' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'lora':
            for name, param in self.named_parameters():
                if 'my_module' not in name:
                    param.requires_grad = self.requires_grad
            for name, m in self.named_modules():  # for lora
                if isinstance(m, MergedLinear):
                    for name, param in m.named_parameters():
                        if "lora_" in name:
                            param.requires_grad = True
        elif self.finetune_mode == 'norm_tuning':
            for name, param in self.named_parameters():
                if 'norm' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'simple_mona':
            self.simple_mona_module1 = Simple_Mona(embed_dims,8)
            self.simple_mona_module2 = Simple_Mona(embed_dims,8)
            for name, param in self.named_parameters():
                if 'simple_mona' not in name:
                    param.requires_grad = self.requires_grad
        ##其他模式则不变

    def forward(self, x, hw_shape,text_features=None,gt_info=None):

        def _inner_forward(x,text_features=None,gt_info=None):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity
            if self.finetune_mode == 'mona':
                x = self.mona_module1(x, hw_shape)
            elif self.finetune_mode == 'odmona':
                x = self.odmona_module1(x, hw_shape)
            elif self.finetune_mode == 'simple_mona':
                x = self.simple_mona_module1(x, hw_shape)
            elif self.finetune_mode == 'seeker':
                x = self.seeker_module1(x, hw_shape, text_features=text_features)
                if isinstance(x, tuple):
                    x, consistency_loss_1 = x
            elif self.finetune_mode == 'visual_seeker':
                x = self.visual_seeker_module1(x, hw_shape, gt_info=gt_info)
            elif self.finetune_mode == 'omni_seeker':
                x = self.omni_seeker_module1(x, hw_shape, text_features=text_features)
            # elif self.finetune_mode == 'adapter':
            #     x = self.adapter_module1(x) #mona原论文里面的adapter位置和这里不一致

            identity = x
            if self.finetune_mode == 'adapter_former':
                adapt_x = self.adapter_former_module1(x)
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            if self.finetune_mode == 'mona':
                x = self.mona_module2(x, hw_shape)
            elif self.finetune_mode == 'odmona':
                x = self.odmona_module2(x, hw_shape)
            elif self.finetune_mode == 'simple_mona':
                x = self.simple_mona_module2(x, hw_shape)
            elif self.finetune_mode == 'seeker':
                x = self.seeker_module2(x, hw_shape,text_features=text_features)
                if isinstance(x, tuple):
                    x, consistency_loss_2 = x
            elif self.finetune_mode == 'visual_seeker':
                x = self.visual_seeker_module2(x, hw_shape, gt_info=gt_info)
            elif self.finetune_mode == 'omni_seeker':
                x = self.omni_seeker_module2(x, hw_shape,text_features=text_features)
            # elif self.finetune_mode == 'adapter':
            #     x = self.adapter_module2(x)

            elif self.finetune_mode == 'adapter_former':
                x = x+adapt_x

            if self.text_consistency_loss:
                return (x, consistency_loss_1+consistency_loss_2)
            else:
                return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward,x,text_features,gt_info)#这里不太确定喵
        else:
            x = _inner_forward(x,text_features,gt_info)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
        requires_grad (bool): Whether to require gradients.
            Default: False.
        finetune_mode (str): The mode of finetuning.
            Default: None.
        text_dim (int, optional): Text dimension. Default: 256.
        text_consistency_loss (bool): Whether to use text consistency loss. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 requires_grad=False,
                 finetune_mode=None,
                 num_prompts=None,
                 prompt_location=None,
                 deep_prompt=None,
                 text_dim=256,
                 text_consistency_loss=False,
                 tokens_of_interest=None,
                 ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        self.text_consistency_loss = text_consistency_loss
        #vpt参数
        self.deep_prompt = deep_prompt
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if self.deep_prompt and self.prompt_location != "prepend":
            raise ValueError("deep prompt mode for swin is only applicable to prepend")
        
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
                requires_grad=requires_grad,
                finetune_mode=finetune_mode,
                num_prompts=num_prompts,
                prompt_location=prompt_location,
                text_dim=text_dim,
                text_consistency_loss=text_consistency_loss,
                tokens_of_interest=tokens_of_interest)
            self.blocks.append(block)

        # self.downsample = downsample
        if downsample is not None:
            if num_prompts is None:
                self.downsample = downsample
            else:
                # downsample is PromptedPatchMerging(PatchMerging)
                self.downsample = downsample
                # add prompt related attributes
                self.downsample.num_prompts = num_prompts
                self.downsample.prompt_location = prompt_location
                if prompt_location == "prepend":
                    if not deep_prompt:
                        self.downsample.prompt_upsampling = None
                        # self.prompt_upsampling = nn.Linear(dim, 4 * dim, bias=False)
                    else:
                        self.downsample.prompt_upsampling = None
        else:
            self.downsample = None
    
    #TODO:这里得改新的forward了
    def forward(self, x, hw_shape,text_features=None,deep_prompt=None,gt_info=None):
        if self.deep_prompt:
            assert deep_prompt is not None
            return self.forward_deep(x, hw_shape, text_features,deep_prompt,gt_info)
        if self.text_consistency_loss:
            consistency_loss = torch.zeros(1, device=x.device)
        for block in self.blocks:
            x = block(x, hw_shape,text_features,gt_info)
            if isinstance(x, tuple) and self.text_consistency_loss:
                x, consistency_loss_i = x
                consistency_loss += consistency_loss_i
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            if self.text_consistency_loss:
                return x_down, down_hw_shape, x, hw_shape, consistency_loss
            else:
                return x_down, down_hw_shape, x, hw_shape
        else:
            if self.text_consistency_loss:
                return x, hw_shape, x, hw_shape, consistency_loss
            else:
                return x, hw_shape, x, hw_shape

    def forward_deep(self, x, hw_shape,text_features=None,deep_prompt=None,gt_info=None):
        # forwards for deep prompt
        assert self.deep_prompt
        # only support prepend
        assert self.prompt_location == "prepend"
        if self.text_consistency_loss:
            consistency_loss = torch.zeros(1, device=x.device)
        # add the prompt embed before each blk call
        B = x.shape[0]  # batchsize
        num_blocks = len(self.blocks)
        if deep_prompt.shape[0] != num_blocks:
            # first layer
            #Swin的第一个stage的deep prompt数量是depths[0] - 1，而不是num_blocks（因为第一个block用shallow prompt，后续每个block前插入deep prompt）
            for i in range(num_blocks):
                if i == 0:
                    x = self.blocks[i](x, hw_shape,text_features,gt_info)

                else:
                    prompt_emb = deep_prompt[i - 1].expand(B, -1, -1)
                    x = torch.cat(
                        (prompt_emb, x[:, self.num_prompts:, :]),
                        dim=1
                    )
                    x = self.blocks[i](x, hw_shape,text_features,gt_info)
                    if isinstance(x, tuple) and self.text_consistency_loss:
                        x, consistency_loss_i = x
                        consistency_loss += consistency_loss_i
        else:
            # other layers
            for i in range(num_blocks):
                prompt_emb = deep_prompt[i].expand(B, -1, -1)
                x = torch.cat(
                    (prompt_emb, x[:, self.num_prompts:, :]),
                    dim=1
                )
                x = self.blocks[i](x, hw_shape,text_features,gt_info)
                if isinstance(x, tuple) and self.text_consistency_loss:
                    x, consistency_loss_i = x
                    consistency_loss += consistency_loss_i
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            if self.text_consistency_loss:
                return x_down, down_hw_shape, x, hw_shape, consistency_loss
            else:
                return x_down, down_hw_shape, x, hw_shape
        else:
            if self.text_consistency_loss:
                return x, hw_shape, x, hw_shape, consistency_loss
            else:
                return x, hw_shape, x, hw_shape

class Adapter(BaseModule):
    def __init__(self,
                 in_dim):
        super().__init__()
        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        project1 = self.project1(x)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)
        return x+project2

class AdapterFormer(BaseModule):
    def __init__(self,
                 in_dim):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)
        self.scale = 0.1
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        project1 = self.project1(x)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)
        project2 = project2*self.scale

        return project2


class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x


class ODConvAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(ODConvAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.norm = nn.GroupNorm(1, attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.norm(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    """ kernel_size = 1 or 3 """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = ODConvAttention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)

class Mona(BaseModule):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(64)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2


class ODMona(BaseModule):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = ODConv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=4, reduction=0.0625, kernel_num=3)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2

class Simple_Mona(BaseModule):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()

        # self.project1 = nn.Linear(in_dim, 64)
        # self.nonlinear = F.gelu
        # self.project2 = nn.Linear(64, in_dim)

        # self.dropout = nn.Dropout(p=0.1)

        # self.adapter_conv = MonaOp(64)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        # identity = x

        x = self.norm(x) * self.gamma + x * self.gammax
        return x 
        # project1 = self.project1(x)

        # b, n, c = project1.shape
        # h, w = hw_shapes
        # project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        # project1 = self.adapter_conv(project1)
        # project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        # nonlinear = self.nonlinear(project1)
        # nonlinear = self.dropout(nonlinear)
        # project2 = self.project2(nonlinear)

        # return identity + project2

#gt转为同层的gt
def get_feature_positions(gt_info, hw_shape):
    """
    计算GT bbox在指定特征层级上的位置
    
    Args:
        gt_info: 预处理后的GT信息
        hw_shape: 特征图的高度和宽度 (H, W)
    
    Returns:
        feature_positions: 特征图上的GT位置信息
    """
    H, W = hw_shape
    feature_positions = []
    
    for batch_idx, (bboxes, num_gt) in enumerate(zip(gt_info['bboxes'], gt_info['num_gt_per_image'])):
        if num_gt == 0:
            feature_positions.append(torch.empty(0, 4))
            continue
            
        # 将归一化坐标转换为特征图坐标
        feature_coords = bboxes.clone()
        feature_coords[:, [0, 2]] *= W  # x坐标乘以特征图宽度
        feature_coords[:, [1, 3]] *= H  # y坐标乘以特征图高度
        
        # 转换为整数坐标
        feature_coords = feature_coords.long()
        feature_positions.append(feature_coords)
    
    return feature_positions

class TopkSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, k):
        topk_idx = torch.topk(logits, k, dim=1)[1]
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, topk_idx, 1.0)
        ctx.save_for_backward(logits)
        return one_hot

    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        soft_grad = torch.softmax(logits, dim=1)
        return grad_output * soft_grad, None

def ste_topk_mask(logits, k):
    return TopkSTE.apply(logits, k)

# 确保两个分支的特征分布相似
def feature_consistency_loss(vision_feat, text_feat):
    return F.mse_loss(
        F.normalize(vision_feat.mean(dim=1), p=2, dim=-1),
        F.normalize(text_feat.squeeze(1), p=2, dim=-1)
    )

from torchvision.ops import roi_align
#TODO:第四处注入，需要形成原型库，使用原型库进行增强
class VisualSeekerAdapter(nn.Module):
    """
    动态寻找稀疏token并进行增强
    """
    def __init__(self,
                 in_dim,
                 down_project_dim=64,#降维维度
                 m=16,#m个query
                 k=64,#topk个token
                 attention_heads=4,
                 dropout_rate=0.1,
                 prototype_update_momentum=0.9,  # 原型更新动量
                 temperature=0.1,  # 相似度温度参数
                 roi_size=(3, 3),  # ROI Align的输出尺寸
                 ):
        super().__init__()
        
        self.m = m
        self.k = k
        self.in_dim = in_dim
        self.projected_dim = down_project_dim
        self.prototype_update_momentum = prototype_update_momentum
        self.temperature = temperature
        self.roi_size = roi_size

        # --- 核心 Adapter 结构 ---
        # 1. 降维
        self.down_project = nn.Linear(in_dim, down_project_dim)
        # 2. 非线性激活与 Dropout (紧跟降维之后)
        self.nonlinear_activation = F.gelu
        self.dropout = nn.Dropout(dropout_rate)
        # 3. 升维
        self.up_project = nn.Linear(down_project_dim, in_dim)
        
        #Query/Prototype 初始化为 nn.Parameter
        self.m_queries = nn.Parameter(torch.randn(1, m, down_project_dim))
        
        # 原型匹配网络：将GT特征投影到query空间
        # self.prototype_matcher = nn.Sequential(
        #     nn.Linear(down_project_dim, down_project_dim),
        #     nn.ReLU(),
        #     nn.Linear(down_project_dim, down_project_dim)
        # )
        # ROI Align层，用于提取GT区域特征
        # self.roi_align = roi_align(
        #     output_size=roi_size,
        #     spatial_scale=1.0,  # 需要根据实际特征图尺寸调整
        #     sampling_ratio=-1,
        #     aligned=True
        # )
        
        # 交互模块: 只有一个自注意力层和 LayerNorm
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=down_project_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(down_project_dim)
        
        # 残差缩放因子
        self.gamma = nn.Parameter(torch.tensor(1e-1))
        # 原型库状态跟踪
        self.prototype_usage_count = nn.Parameter(torch.zeros(m), requires_grad=False)

    def extract_gt_features(self, activated_features, hw_shapes,gt_info):
        """
        从激活后的特征中提取GT区域特征
        
        Args:
            activated_features: 激活后的特征 [B, N, C_proj]
            gt_info: GT信息字典，包含bboxes, labels等
            hw_shape: 特征图尺寸 (H, W)
        
        Returns:
            gt_features: GT特征列表 [B个元素，每个是[num_gt, C_proj]]
            gt_labels: GT标签列表 [B个元素，每个是[num_gt]]
        """
        B, N, C_proj = activated_features.shape
        H, W = hw_shapes
        
        # 获取GT在特征图上的位置
        feature_positions = get_feature_positions(gt_info, hw_shapes)
        
        gt_features = []
        gt_labels = []
        
        for batch_idx in range(B):
            batch_features = activated_features[batch_idx]  # [N, C_proj]
            batch_positions = feature_positions[batch_idx]  # [num_gt, 4]
            batch_labels = gt_info['labels'][batch_idx]  # [num_gt]
            
            if len(batch_positions) == 0:
                gt_features.append(torch.empty(0, C_proj))
                gt_labels.append(torch.empty(0, dtype=batch_labels.dtype))
                continue
            
            # 将特征重塑为空间形式 [H, W, C_proj]
            spatial_features = batch_features.view(H, W, C_proj).permute(2, 0, 1)  # [C_proj, H, W]
            
            # 准备ROI Align的输入
            # 需要将特征图坐标转换为ROI Align期望的格式
            rois = []
            valid_gt_indices = []
            
            for gt_idx, bbox in enumerate(batch_positions):
                x1, y1, x2, y2 = bbox.float()
                
                # 确保坐标在有效范围内
                x1 = torch.clamp(x1, 0, W)
                y1 = torch.clamp(y1, 0, H)
                x2 = torch.clamp(x2, 0, W)
                y2 = torch.clamp(y2, 0, H)
                
                # 检查bbox是否有效
                if x2 <= x1 or y2 <= y1:
                    continue
                 # ROI Align期望的格式：[batch_idx, x1, y1, x2, y2]
                roi = torch.tensor([0, x1, y1, x2, y2], dtype=torch.float32, device=spatial_features.device)
                rois.append(roi)
                valid_gt_indices.append(gt_idx)
            if not rois:
                gt_features.append(torch.empty(0, C_proj))
                gt_labels.append(torch.empty(0, dtype=batch_labels.dtype))
                continue
            
            # 转换为tensor
            rois = torch.stack(rois)  # [num_valid_gt, 5]
            valid_labels = batch_labels[valid_gt_indices]
            
            # 使用ROI Align提取特征
            # 注意：ROI Align期望输入是[B, C, H, W]格式
            spatial_features = spatial_features.unsqueeze(0)  # [1, C_proj, H, W]
            
            try:
                roi_features = roi_align(
                    spatial_features,  # [1, C_proj, H, W]
                    rois,             # [num_valid_gt, 4] - [x1, y1, x2, y2]
                    output_size=self.roi_size,  # (3, 3)
                    spatial_scale=1.0,  # 需要根据实际特征图尺寸调整
                    sampling_ratio=-1,
                    aligned=True
                )  # [num_valid_gt, C_proj, roi_h, roi_w]
                
                # 全局平均池化得到特征向量
                roi_features = roi_features.mean(dim=(2, 3))  # [num_valid_gt, C_proj]
                
                gt_features.append(roi_features)
                gt_labels.append(valid_labels)
                
            except Exception as e:
                # 如果ROI Align失败，使用中心点特征作为备选方案
                print(f"ROI Align failed: {e}, using center point features")
                center_features = []
                for bbox in batch_positions[valid_gt_indices]:
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    # 使用clamp确保中心点坐标在有效范围内
                    center_x = torch.clamp(center_x, 0, W)
                    center_y = torch.clamp(center_y, 0, H)
                
                    center_feature = batch_features[center_y * W + center_x]  # [C_proj]
                    center_features.append(center_feature)
                
                if center_features:
                    center_features = torch.stack(center_features)  # [num_valid_gt, C_proj]
                    gt_features.append(center_features)
                    gt_labels.append(valid_labels)
                else:
                    gt_features.append(torch.empty(0, C_proj))
                    gt_labels.append(torch.empty(0, dtype=batch_labels.dtype))
        
        return gt_features, gt_labels

    def update_prototypes_with_gt(self, activated_features,hw_shapes,gt_info):
        """
        基于GT信息更新query/prototype
        """
        if gt_info is None:
            return
            
        gt_features, gt_labels = self.extract_gt_features(activated_features, hw_shapes, gt_info)
        
        # 更清晰的empty情况处理
        if not gt_features:
            return
        
        # 过滤掉empty的特征和标签
        valid_features = []
        valid_labels = []
        
        for features, labels in zip(gt_features, gt_labels):
            if len(features) > 0 and len(labels) > 0:
                valid_features.append(features)
                valid_labels.append(labels)
        
        # 如果没有有效的GT特征，直接返回
        if not valid_features:
            return
            
        # 拼接所有有效的GT特征
        all_gt_features = torch.cat(valid_features, dim=0)  # [total_valid_gt, C_proj]
        all_gt_labels = torch.cat(valid_labels, dim=0)  # [total_valid_gt]
        
        # 按类别分组更新原型
        unique_labels = torch.unique(all_gt_labels)
        
        for label in unique_labels:
            label_mask = (all_gt_labels == label)
            label_features = all_gt_features[label_mask]  # [num_gt_class, C_proj]
            
            if len(label_features) == 0:
                continue
                
            # 计算该类别的平均特征
            avg_gt_feature = label_features.mean(dim=0)  # [C_proj]
            
            # 计算与现有原型的相似度
            current_prototypes = self.m_queries.data.squeeze(0)  # [m, C_proj]
            similarities = F.cosine_similarity(
                avg_gt_feature.unsqueeze(0), current_prototypes, dim=-1
            )  # [m]
            
            # 找到最相似的原型
            most_similar_idx = similarities.argmax()
            similarity_score = similarities[most_similar_idx]
            
            # 如果相似度足够高，更新该原型
            if similarity_score > 0.5:  # 相似度阈值
                with torch.no_grad():
                    # 使用动量更新
                    self.m_queries.data[0, most_similar_idx] = (
                        self.prototype_update_momentum * self.m_queries.data[0, most_similar_idx] +
                        (1 - self.prototype_update_momentum) * avg_gt_feature
                    )
                    # 更新使用计数
                    self.prototype_usage_count[most_similar_idx] += 1

    def select_tokens_with_prototypes(self, activated_features):
        """
        基于原型特征选择最相关的tokens
        
        Args:
            activated_features: 激活后的特征 [B, N, C_proj]
        
        Returns:
            topk_indices: topk索引 [B, k]
            sparse_image_tokens: 选中的稀疏tokens [B, k, C_proj]
        """
        B, N, C_proj = activated_features.shape
        
        # 计算图像特征与原型query的相似度
        norm_image_features = F.normalize(activated_features, p=2, dim=-1)  # [B, N, C_proj]
        norm_queries = F.normalize(self.m_queries.expand(B, -1, -1), p=2, dim=-1)  # [B, m, C_proj]
        
        # 计算相似度矩阵
        similarities = torch.bmm(norm_image_features, norm_queries.transpose(1, 2))  # [B, N, m]
        
        # 对每个原型，选择最相似的tokens
        all_scores = []
        for b in range(B):
            batch_similarities = similarities[b]  # [N, m]
            # 取每个原型对应的最大相似度
            max_similarities, _ = batch_similarities.max(dim=1)  # [N]
            all_scores.append(max_similarities)
        
        all_scores = torch.stack(all_scores)  # [B, N]
        
        # 使用STE生成topk mask
        mask = ste_topk_mask(all_scores, self.k)  # [B, N]
        
        # 获取topk的索引
        topk_indices = mask.nonzero(as_tuple=False).view(B, self.k, 2)[:, :, 1]  # [B, k]
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.projected_dim)
        
        # 提取稀疏tokens
        sparse_image_tokens = torch.gather(activated_features, 1, topk_indices_expanded)  # [B, k, C_proj]
        
        return topk_indices, sparse_image_tokens, topk_indices_expanded

    def forward(self, image_features, hw_shapes=None, gt_info=None):
        """
        Args:
            image_features: 输入特征 [B, N, C]
            hw_shapes: 特征图尺寸列表
            gt_info: GT信息字典，包含gt_features等
        """
        B, N, C_in = image_features.shape
        identity = image_features

        # --- 1. 降维和激活 ---
        projected_features = self.down_project(image_features)
        activated_features = self.dropout(self.nonlinear_activation(projected_features))  # [B, N, C_proj]

        # --- 2. 基于GT信息更新原型 ---
        if gt_info is not None and hw_shapes is not None:
            self.update_prototypes_with_gt(activated_features, hw_shapes, gt_info)

        # --- 3. 基于原型选择tokens ---
        topk_indices, sparse_image_tokens, topk_indices_expanded = self.select_tokens_with_prototypes(activated_features)

        # --- 4. 原型与稀疏tokens的交互 ---
        queries = self.m_queries.expand(B, -1, -1)  # [B, m, C_proj]
        combined_sequence = torch.cat([queries, sparse_image_tokens], dim=1)  # [B, m+k, C_proj]
        
        # 交互: Norm -> Self-Attention -> Add
        attention_input = self.norm(combined_sequence)
        attention_output, _ = self.interaction_attention(
            query=attention_input, key=attention_input, value=attention_input
        )
        enhanced_sequence = combined_sequence + attention_output
        
        # 分离出增强后的原型和稀疏tokens
        enhanced_queries = enhanced_sequence[:, :self.m, :]  # [B, m, C_proj]
        enhanced_sparse_tokens = enhanced_sequence[:, self.m:, :]  # [B, k, C_proj]

        # --- 5. 原型更新（EMA平滑）---
        with torch.no_grad():
            # 使用增强后的原型更新query参数
            self.m_queries.copy_(
                self.prototype_update_momentum * self.m_queries + 
                (1 - self.prototype_update_momentum) * enhanced_queries.mean(dim=0, keepdim=True)
            )
        
        # --- 6. 信息还原与升维 ---
        # 创建零张量并填充增强后的tokens
        delta_x_projected = torch.zeros_like(activated_features)
        delta_x_projected.scatter_(1, topk_indices_expanded, enhanced_sparse_tokens)
        
        # 升维
        delta_x = self.up_project(delta_x_projected)  # [B, N, C_in]
        
        return identity + delta_x * self.gamma


class DynamicSeekerAdapter(nn.Module):
    """
    动态寻找稀疏token并进行增强
    """
    def __init__(self,
                 in_dim,
                 text_dim,
                 down_project_dim=64,#降维维度
                 m=16,#m个query
                 k=64,#topk个token
                 attention_heads=4,
                 dropout_rate=0.1,
                 text_consistency_loss=False,
                 tokens_of_interest=None,#可变类型，比如list[0]或者str 'all'
                 ):
        super().__init__()
        
        self.m = m
        self.k = k
        self.in_dim = in_dim
        self.projected_dim = down_project_dim
        self.text_consistency_loss = text_consistency_loss
        self.tokens_of_interest = tokens_of_interest

        # --- 核心 Adapter 结构 ---
        # 1. 降维
        self.down_project = nn.Linear(in_dim, down_project_dim)
        # 2. 非线性激活与 Dropout (紧跟降维之后)
        self.nonlinear_activation = F.gelu
        self.dropout = nn.Dropout(dropout_rate)
        # 3. 升维
        self.up_project = nn.Linear(down_project_dim, in_dim)
        # --- 结构结束 ---
        
        
        # Query 初始化为 nn.Parameter
        self.m_queries = nn.Parameter(torch.randn(1, m, down_project_dim))
        
        # 文本处理器 (简化版)
        if text_consistency_loss:
            self.text_projector = nn.Linear(text_dim, down_project_dim) #之前被压缩的
        
        # 交互模块: 只有一个自注意力层和 LayerNorm
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=down_project_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(down_project_dim)
        
        # 残差缩放因子
        self.gamma = nn.Parameter(torch.tensor(1e-1))



    def forward(self, image_features, hw_shapes=None, text_features=None):
        assert text_features is not None, "text_features 不能为 None"
        B, N, C_in = image_features.shape
        identity = image_features

        # --- 1. 对输入特征应用 Adapter 前半部分：降维 -> GELU -> Dropout ---
        projected_features = self.down_project(image_features)
        activated_features = self.dropout(self.nonlinear_activation(projected_features)) # (B, N, C_proj)

        # --- 2. 文本特征处理 ---
        if self.text_consistency_loss:
            # 1) 所有token都过projector
            projected_text_feat = self.text_projector(text_features)  # (B, L, C_proj)
            # 2) consistency_loss只用CLS
            projected_text_feat_for_loss = projected_text_feat[:, 0, :].unsqueeze(1)  # (B, 1, C_proj)
            consistency_loss = feature_consistency_loss(activated_features, projected_text_feat_for_loss)
        else:
            # 不做投影，直接截断
            projected_text_feat = text_features[:, :, :self.projected_dim]  # (B, L, C_proj)
            consistency_loss = None

        # --- 3. 处理tokens_of_interest ---
        tokens_of_interest = self.tokens_of_interest
        if tokens_of_interest is None:
            # 只用CLS
            selected_text_feat = projected_text_feat[:, 0, :].unsqueeze(1)  # (B, 1, C_proj)
            norm_key_text_tokens = F.normalize(selected_text_feat, p=2, dim=-1)
            image_scores = torch.bmm(F.normalize(activated_features, p=2, dim=-1), norm_key_text_tokens.transpose(1, 2)).squeeze(-1)  # (B, N)
        elif tokens_of_interest == 'all':
            # 用所有token
            selected_text_feat = projected_text_feat  # (B, L, C_proj)
            norm_key_text_tokens = F.normalize(selected_text_feat, p=2, dim=-1)
            image_scores = torch.bmm(F.normalize(activated_features, p=2, dim=-1), norm_key_text_tokens.transpose(1, 2))  # (B, N, L)
            image_scores, _ = image_scores.max(dim=2)  # (B, N)
        else:
            # 用指定token
            if isinstance(tokens_of_interest, int):
                idx = tokens_of_interest
                selected_text_feat = projected_text_feat[:, idx, :].unsqueeze(1)  # (B, 1, C_proj)
                norm_key_text_tokens = F.normalize(selected_text_feat, p=2, dim=-1)
                image_scores = torch.bmm(F.normalize(activated_features, p=2, dim=-1), norm_key_text_tokens.transpose(1, 2)).squeeze(-1)  # (B, N)
            elif isinstance(tokens_of_interest, (list, tuple)):
                idx = tokens_of_interest
                selected_text_feat = projected_text_feat[:, idx, :]  # (B, L', C_proj)
                norm_key_text_tokens = F.normalize(selected_text_feat, p=2, dim=-1)
                image_scores = torch.bmm(F.normalize(activated_features, p=2, dim=-1), norm_key_text_tokens.transpose(1, 2))  # (B, N, L')
                image_scores, _ = image_scores.max(dim=2)  # (B, N)
            else:
                raise ValueError("token_of_interest必须为None、'all'、int或list/tuple")
        # --- 2. 文本引导与 Top-k 选择 (在激活后的特征上进行) ---
        # 替代方案：只用CLS token
        # pooled_text_feat = text_features[:,0] #使用CLS token
        # if self.text_consistency_loss:
        #     projected_text_feat = self.text_projector(pooled_text_feat).unsqueeze(1)
        #     consistency_loss = feature_consistency_loss(activated_features, projected_text_feat)
        # # 现在权宜方案，直接截断取前64维
        # else:
        #     projected_text_feat = pooled_text_feat[:, :self.projected_dim].unsqueeze(1)
        # norm_image_features = F.normalize(activated_features, p=2, dim=-1)
        # norm_key_text_tokens = F.normalize(projected_text_feat, p=2, dim=-1)
        # image_scores = torch.bmm(norm_image_features, norm_key_text_tokens.transpose(1, 2)).squeeze(-1)

        # --- 2. 文本引导与 Top-k 选择 (在激活后的特征上进行) ---
        # 替代方案：使用所有token
        # projected_text_feat = text_features[:, :, :self.projected_dim]  # (B, L, C_proj)
        # norm_image_features = F.normalize(activated_features, p=2, dim=-1)  # (B, N, C_proj)
        # norm_key_text_tokens = F.normalize(projected_text_feat, p=2, dim=-1)  # (B, L, C_proj)
        # # 计算所有图像token与所有文本token的相似度
        # image_scores = torch.bmm(norm_image_features, norm_key_text_tokens.transpose(1, 2))  # (B, N, L)
        # # 在文本token维度取最大值
        # image_scores, _ = image_scores.max(dim=2)  # (B, N)
        
        # --------- 用STE生成mask ---------
        mask = ste_topk_mask(image_scores, self.k)  # (B, N)
        # 获取topk的索引
        topk_indices = mask.nonzero(as_tuple=False).view(B, self.k, 2)[:,:,1]  # (B, k)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.projected_dim)
        sparse_image_tokens = torch.gather(activated_features, 1, topk_indices_expanded)

        # --- 4. 简化版注意力交互 ---
        queries = self.m_queries.expand(B, -1, -1)
        combined_sequence = torch.cat([queries, sparse_image_tokens], dim=1)
        
        # 交互: Norm -> Self-Attention -> Add
        attention_input = self.norm(combined_sequence)
        attention_output, _ = self.interaction_attention(
            query=attention_input, key=attention_input, value=attention_input
        )
        # 只有一个残差连接，没有后续 FFN
        enhanced_sequence = combined_sequence + attention_output
        
        # 分离出增强后的图像 token和query
        enhanced_queries = enhanced_sequence[:, :self.m, :]  # (B, m, C_proj)
        enhanced_sparse_tokens = enhanced_sequence[:, self.m:, :]

        # 替换原本的 query 参数，使用不带梯度的EMA平滑更新
        with torch.no_grad():
            self.m_queries.copy_(0.8 * self.m_queries + 0.2 * enhanced_queries.mean(dim=0, keepdim=True))
        
        # --- 4. 信息还原与升维 ---
        # 创建一个与激活后特征图形状相同的零张量
        delta_x_projected = torch.zeros_like(activated_features)
        delta_x_projected.scatter_(1, topk_indices_expanded, enhanced_sparse_tokens)
        
        # 应用 Adapter 后半部分：升维
        delta_x = self.up_project(delta_x_projected) # -> (B, N, C_in)
        
        if self.text_consistency_loss:
            return identity + delta_x * self.gamma, consistency_loss*10
        else:
            return identity + delta_x * self.gamma

class OmniDynamicSeekerAdapter(nn.Module):
    """
    动态寻找稀疏token并进行增强，区别是将文本和视觉对齐到了同一个空间
    """
    def __init__(self,
                 in_dim,
                 text_dim,
                 down_project_dim=64,#降维维度
                 m=16,#m个query
                 k=64,#topk个token
                 attention_heads=4,
                 dropout_rate=0.1,
                 ):
        super().__init__()
        
        self.m = m
        self.k = k
        self.in_dim = in_dim
        self.projected_dim = down_project_dim

        # --- 核心 Adapter 结构 ---
        # 1. 降维
        self.down_project = nn.Linear(in_dim,text_dim)
        self.omni_down_project = nn.Linear(text_dim, down_project_dim)
        # 2. 非线性激活与 Dropout (紧跟降维之后)
        self.nonlinear_activation = F.gelu
        self.dropout = nn.Dropout(dropout_rate)
        # 3. 升维
        self.up_project = nn.Linear(down_project_dim, in_dim)
        # --- 结构结束 ---
        
        # Query 初始化为 nn.Parameter
        self.m_queries = nn.Parameter(torch.randn(1, m, down_project_dim))
        
        # 交互模块: 只有一个自注意力层和 LayerNorm
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=down_project_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(down_project_dim)
        
        # 残差缩放因子
        self.gamma = nn.Parameter(torch.tensor(1e-1))



    def forward(self, image_features, hw_shapes=None,text_features=None):
        assert text_features is not None, "text_features 不能为 None"
        #TODO：1.实现辅助loss 2.增加修改输出为维度（）
        B, N, C_in = image_features.shape
        identity = image_features
        
        # --- 1. 对输入特征应用 Adapter 前半部分：降维 -> GELU -> Dropout ---
        projected_features = self.down_project(image_features) #到text_dim了
        projected_features = self.dropout(self.nonlinear_activation(projected_features))
        activated_features = self.omni_down_project(projected_features) # -> (B, N, C_proj)

        # --- 2. 文本引导与 Top-k 选择 (在激活后的特征上进行) ---
        # #原来的
        pooled_text_feat = text_features[:,0] #使用CLS token
        projected_text_feat = self.omni_down_project(pooled_text_feat).unsqueeze(1)
        eps = 1e-8
        norm_image_features = F.normalize(activated_features+eps, p=2, dim=-1)
        norm_key_text_tokens = F.normalize(projected_text_feat+eps, p=2, dim=-1)
        image_scores = torch.bmm(norm_image_features, norm_key_text_tokens.transpose(1, 2)).squeeze(-1)

        
        # --------- 用STE生成mask ---------
        mask = ste_topk_mask(image_scores, self.k)  # (B, N)
        # 获取topk的索引
        topk_indices = mask.nonzero(as_tuple=False).view(B, self.k, 2)[:,:,1]  # (B, k)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.projected_dim)
        sparse_image_tokens = torch.gather(activated_features, 1, topk_indices_expanded)

        # --- 3. 简化版注意力交互 ---
        queries = self.m_queries.expand(B, -1, -1)
        combined_sequence = torch.cat([queries, sparse_image_tokens], dim=1)
        
        # 交互: Norm -> Self-Attention -> Add
        attention_input = self.norm(combined_sequence)
        attention_output, _ = self.interaction_attention(
            query=attention_input, key=attention_input, value=attention_input
        )
        # 只有一个残差连接，没有后续 FFN
        enhanced_sequence = combined_sequence + attention_output
        
        # 分离出增强后的图像 token和query
        enhanced_queries = enhanced_sequence[:, :self.m, :]  # (B, m, C_proj)
        enhanced_sparse_tokens = enhanced_sequence[:, self.m:, :]

        # 替换原本的 query 参数，使用不带梯度的EMA平滑更新
        with torch.no_grad():
            self.m_queries.copy_(0.8 * self.m_queries + 0.2 * enhanced_queries.mean(dim=0, keepdim=True))
        
        # --- 4. 信息还原与升维 ---
        # 创建一个与激活后特征图形状相同的零张量
        delta_x_projected = torch.zeros_like(activated_features)
        delta_x_projected.scatter_(1, topk_indices_expanded, enhanced_sparse_tokens)
        
        # 应用 Adapter 后半部分：升维
        delta_x = self.up_project(delta_x_projected) # -> (B, N, C_in)
        
        return identity + delta_x * self.gamma




@MODELS.register_module()
class SwinTransformer(BaseModule):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
        requires_grad (bool): Whether to require gradients.
            Default: False.
        finetune_mode (str): The mode of finetuning.
            Default: None.
        prompt_cfg (dict, optional): The Config for prompt tuning.
            Default: None. #用于VPT模式
        text_dim (int): 文本维度,默认256,用于seeker_adapter
        text_consistency_loss (bool): 是否使用文本一致性损失,用于seeker_adapter
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
                 init_cfg=None,
                 requires_grad=False,
                 finetune_mode=None,
                 prompt_cfg=None,
                 text_dim=256,
                 text_consistency_loss=False,
                 tokens_of_interest=None,
                 ):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SwinTransformer, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        # VPT模式相关
        self.is_vpt = (finetune_mode == "vpt" and prompt_cfg is not None)

        self.prompt_num_tokens = None
        self.prompt_location = None
        self.prompt_deep = None
        if self.is_vpt:
            patch_size_tuple = to_2tuple(patch_size)
            self.prompt_cfg = prompt_cfg
            self.prompt_num_tokens = prompt_cfg.num_tokens
            self.prompt_location = prompt_cfg.location
            self.prompt_deep = prompt_cfg.deep
            self.prompt_dropout = prompt_cfg.dropout
            self.prompt_initiation = prompt_cfg.initiation
            self.prompt_dropout_layer = nn.Dropout(self.prompt_dropout)
            self.prompt_proj = nn.Identity()
            # prompt embedding初始化
            if self.prompt_initiation == "random":
                val = math.sqrt(6. / float(3 * reduce(mul, patch_size_tuple, 1) + embed_dims))
                assert self.prompt_location == 'prepend'
                self.vpt_prompt_embeddings = nn.Parameter(torch.zeros(
                    1, self.prompt_num_tokens, embed_dims))
                nn.init.uniform_(self.vpt_prompt_embeddings.data, -val, val)
                if self.prompt_deep:
                    self.vpt_deep_prompt_embeddings = nn.ParameterList()
                    for i in range(num_layers):
                        if i == 0:
                            d = depths[0] - 1
                        else:
                            d = depths[i]
                        dim = embed_dims * (2 ** i)
                        deep_emb = nn.Parameter(torch.zeros(d, self.prompt_num_tokens, dim))
                        nn.init.uniform_(deep_emb.data, -val, val)
                        self.vpt_deep_prompt_embeddings.append(deep_emb)
                        #TODO:这里潜在要求第一个block的depth应该不能低于1
            else:
                raise ValueError("Other initiation scheme is not supported")
        #这里初始化text_consistency_loss
        self.text_consistency_loss = text_consistency_loss
        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
                requires_grad=requires_grad,
                finetune_mode=finetune_mode,
                num_prompts=self.prompt_num_tokens,
                prompt_location=self.prompt_location,
                deep_prompt=self.prompt_deep,
                text_dim=text_dim,
                text_consistency_loss=text_consistency_loss,
                tokens_of_interest=tokens_of_interest,
                )
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
        # 其余finetune_mode参数设置保持原有
        if not self.is_vpt:
            if finetune_mode is None:
                for name, param in self.named_parameters():
                    param.requires_grad = requires_grad
            elif finetune_mode == 'mona':
                for name, param in self.named_parameters():
                    if 'mona_module' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'odmona':
                for name, param in self.named_parameters():
                    if 'odmona_module' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'adapter':
                for name, param in self.named_parameters():
                    if 'adapter' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'adapter_former':
                for name, param in self.named_parameters():
                    if 'adapter_former_module' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'bitfit':
                for name, param in self.named_parameters():
                    if 'bias' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'lora':
                for name, param in self.named_parameters():
                    if 'my_module' not in name:
                        param.requires_grad = requires_grad
                for name, m in self.named_modules():  # for lora
                    if isinstance(m, MergedLinear):
                        for name, param in m.named_parameters():
                            if "lora_" in name:
                                param.requires_grad = True
            elif finetune_mode == 'norm_tuning':
                for name, param in self.named_parameters():
                    if 'norm' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'simple_mona':
                for name, param in self.named_parameters():
                    if 'simple_mona' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'seeker':
                for name, param in self.named_parameters():
                    if 'seeker_module' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'visual_seeker':
                for name, param in self.named_parameters():
                    if 'visual_seeker_module' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'omni_seeker':
                for name, param in self.named_parameters():
                    if 'omni_seeker_module' not in name:
                        param.requires_grad = requires_grad
        if self.is_vpt:
            for name, param in self.named_parameters():
                if 'vpt' not in name:
                    param.requires_grad = requires_grad

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    #TODO：第三处注入，需要考虑对有gt的兼容性
    def forward(self, x,text_features=None,gt_info=None):
        if self.text_consistency_loss:
            consistency_loss = torch.zeros(1, device=x.device)
        if getattr(self, 'is_vpt', False):
            # VPT模式
            x, hw_shape = self.patch_embed(x)
            if self.use_abs_pos_embed:
                x = x + self.absolute_pos_embed
            x = self.drop_after_pos(x)
            B = x.shape[0]
            # prepend prompt
            prompt_embed = self.prompt_dropout_layer(self.vpt_prompt_embeddings.expand(B, -1, -1))
            x = torch.cat((prompt_embed, x), dim=1)
            outs = []
            if not self.prompt_deep:
                for i, stage in enumerate(self.stages):
                    if self.text_consistency_loss:
                        x, hw_shape, out, out_hw_shape, consistency_loss_i = stage(x, hw_shape,text_features)
                        consistency_loss += consistency_loss_i
                    else:
                        x, hw_shape, out, out_hw_shape = stage(x, hw_shape,text_features)
                    if i in self.out_indices:
                        norm_layer = getattr(self, f'norm{i}')
                        out = norm_layer(out)
                        # remove prompts
                        out = out[:, self.prompt_num_tokens:, :]
                        out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                        outs.append(out)
            else:
                for i, (stage, deep_prompt_emb) in enumerate(zip(self.stages, self.vpt_deep_prompt_embeddings)):
                    if self.text_consistency_loss:
                        x, hw_shape, out, out_hw_shape, consistency_loss_i = stage(x, hw_shape,text_features,deep_prompt_emb)
                        consistency_loss += consistency_loss_i
                    else:
                        x, hw_shape, out, out_hw_shape = stage(x, hw_shape,text_features,deep_prompt_emb)
                    if i in self.out_indices:
                        norm_layer = getattr(self, f'norm{i}')
                        out = norm_layer(out)
                        # remove prompts
                        out = out[:, self.prompt_num_tokens:, :]
                        out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                        outs.append(out)
            if self.text_consistency_loss:
                return outs, consistency_loss
            else:
                return outs
        else:
            # 原有SwinTransformer逻辑
            x, hw_shape = self.patch_embed(x)
            if self.use_abs_pos_embed:
                x = x + self.absolute_pos_embed
            x = self.drop_after_pos(x)
            outs = []
            for i, stage in enumerate(self.stages):
                if self.text_consistency_loss:
                    x, hw_shape, out, out_hw_shape, consistency_loss_i = stage(x, hw_shape,text_features)
                    consistency_loss += consistency_loss_i
                else:
                    x, hw_shape, out, out_hw_shape = stage(x, hw_shape,text_features)
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    out = norm_layer(out)
                    out = out.view(-1, *out_hw_shape,
                                   self.num_features[i]).permute(0, 3, 1,
                                                                 2).contiguous()
                    outs.append(out)
            if self.text_consistency_loss:
                return outs, consistency_loss
            else:
                return outs


def swin_converter(ckpt):

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt['backbone.' + new_k] = new_v

    return new_ckpt
