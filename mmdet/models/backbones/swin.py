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
        tokens_of_interest (list): 用于obj_seeker的tokens_of_interest. Default: None.
        esod_loss (bool): 是否使用ESOD损失,用于obj_seeker. Default: False.
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
                 tokens_of_interest=None,
                 esod_loss=False,
                 ):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp
        self.text_consistency_loss = text_consistency_loss
        self.esod_loss = esod_loss
        lora_mode = False
        if finetune_mode == 'lora':
            lora_mode = True
        adapter_mode = False
        if finetune_mode == 'adapter':
            adapter_mode = True
        # hyper_mode = False
        # if finetune_mode == 'hyperadapter':
        #     hyper_mode = True
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
        elif self.finetune_mode == 'query_seeker':
            self.query_seeker_module1 = QuerySeekerAdapter(embed_dims,text_dim)
            self.query_seeker_module2 = QuerySeekerAdapter(embed_dims,text_dim)
            for name, param in self.named_parameters():
                if 'query_seeker_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'esod_query_seeker':
            self.esod_query_seeker_module1 = ESODQuerySeekerAdapter(embed_dims,text_dim)
            self.esod_query_seeker_module2 = ESODQuerySeekerAdapter(embed_dims,text_dim)
            for name, param in self.named_parameters():
                if 'esod_query_seeker_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'obj_seeker': #来自ESOD的有gt的模块
            self.obj_seeker_module1 = ObjSeeker(embed_dims)
            self.obj_seeker_module2 = ObjSeeker(embed_dims)
            for name, param in self.named_parameters():
                if 'obj_seeker_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'visual_seeker':
            self.visual_seeker_module1 = VisualSeekerAdapter(embed_dims)
            self.visual_seeker_module2 = VisualSeekerAdapter(embed_dims)
            for name, param in self.named_parameters():
                if 'visual_seeker_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'classaware_visual_seeker':
            self.classaware_visual_seeker_module1 = ClassawareVisualSeekerAdapter(embed_dims)
            self.classaware_visual_seeker_module2 = ClassawareVisualSeekerAdapter(embed_dims)
            for name, param in self.named_parameters():
                if 'classaware_visual_seeker_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'omni_seeker':
            self.omni_seeker_module1 = OmniDynamicSeekerAdapter(embed_dims,text_dim)
            self.omni_seeker_module2 = OmniDynamicSeekerAdapter(embed_dims,text_dim)
            for name, param in self.named_parameters():
                if 'omni_seeker_module' not in name:
                    param.requires_grad = self.requires_grad

        elif self.finetune_mode == 'vfmadapter':
            self.vfmadapter_attn_adapter = VFMAdapter(embed_dims)
            self.vfmadapter_ffn_adapter = VFMAdapter(embed_dims)
            self.vfmadapter_query_projector = nn.Linear(embed_dims, 64) #发现得集中到96
            self.vfmadapter_attn_queries = nn.Parameter(torch.randn(1, 16, 64))
            self.vfmadapter_ffn_queries = nn.Parameter(torch.randn(1, 16, 64))
            for name, param in self.named_parameters():
                if 'vfmadapter' not in name:
                    param.requires_grad = self.requires_grad

        elif self.finetune_mode == 'hyperadapter':
            self.hyperadapter_module1 = HyperAdapter(embed_dims)
            self.hyperadapter_module2 = HyperAdapter(embed_dims)
            for name, param in self.named_parameters():
                if 'hyperadapter_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'multiscale_hyperadapter':
            self.multiscale_hyperadapter_module1 = MultiscaleHyperAdapter(embed_dims)
            self.multiscale_hyperadapter_module2 = MultiscaleHyperAdapter(embed_dims)
            for name, param in self.named_parameters():
                if 'multiscale_hyperadapter_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'hyperadapter_vl':
            # 不开文本损失，只开文本输入
            self.hyperadapter_vl_module1 = HyperAdapterVL(embed_dims)
            self.hyperadapter_vl_module2 = HyperAdapterVL(embed_dims)
            for name, param in self.named_parameters():
                if 'hyperadapter_vl_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'hyperadapter_esod_seeker':
            self.hyperadapter_esod_seeker_module1 = HyperAdapterESODSeeker(embed_dims)
            self.hyperadapter_esod_seeker_module2 = HyperAdapterESODSeeker(embed_dims)
            for name, param in self.named_parameters():
                if 'hyperadapter_esod_seeker_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'hyperadapter_multi':
            self.hyperadapter_multi_module1 = HyperAdapterMulti(embed_dims)
            self.hyperadapter_multi_module2 = HyperAdapterMulti(embed_dims)
            for name, param in self.named_parameters():
                if 'hyperadapter_multi_module' not in name:
                    param.requires_grad = self.requires_grad
        elif self.finetune_mode == 'hyperadapter_mona':
            self.hyperadapter_mona_module1 = HyperAdapterMona(embed_dims)
            self.hyperadapter_mona_module2 = HyperAdapterMona(embed_dims)
            for name, param in self.named_parameters():
                if 'hyperadapter_mona_module' not in name:
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

    def forward(self, x, hw_shape,text_features=None,gt_info=None, hyper_generator=None):

        def _inner_forward(x,text_features=None,gt_info=None,hyper_generator=None):
            identity = x
            x = self.norm1(x)
            if self.finetune_mode == 'vfmadapter':
                tmpx = self.vfmadapter_query_projector(x)
                q = self.vfmadapter_attn_queries.expand(tmpx.shape[0], -1, -1)
                sim = torch.matmul(q, tmpx.transpose(1, 2))
                attn = F.softmax(sim, dim=-1)
                z_per_query = torch.matmul(attn, tmpx)
                z = z_per_query.mean(dim=1)
                tmpx =  self.vfmadapter_attn_adapter(x,z,hyper_generator, hw_shape)
            x = self.attn(x, hw_shape)
            x = x + identity
            if self.finetune_mode == 'vfmadapter':
                x = x + tmpx
            
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
            elif self.finetune_mode == 'query_seeker':
                x = self.query_seeker_module1(x, hw_shape, text_features=text_features)
                if isinstance(x, tuple):
                    x, aux_loss_1 = x
            elif self.finetune_mode == 'esod_query_seeker':
                x = self.esod_query_seeker_module1(x, hw_shape, text_features=text_features,gt_info=gt_info)
                if isinstance(x, tuple):
                    x, aux_loss_1 = x
            elif self.finetune_mode == 'obj_seeker': #来自ESOD的有gt的模块
                x = self.obj_seeker_module1(x,hw_shape,gt_info=gt_info)
                if isinstance(x, tuple):
                    x, seg_loss_1 = x
            elif self.finetune_mode == 'visual_seeker':
                x = self.visual_seeker_module1(x, hw_shape, gt_info=gt_info)
            elif self.finetune_mode == 'classaware_visual_seeker':
                x = self.classaware_visual_seeker_module1(x, hw_shape, gt_info=gt_info)
            elif self.finetune_mode == 'omni_seeker':
                x = self.omni_seeker_module1(x, hw_shape, text_features=text_features)
            elif self.finetune_mode == 'hyperadapter':
                x = self.hyperadapter_module1(x, hyper_generator, hw_shape)
            elif self.finetune_mode == 'multiscale_hyperadapter':
                x = self.multiscale_hyperadapter_module1(x, hyper_generator, hw_shape)
            elif self.finetune_mode == 'hyperadapter_vl':
                x = self.hyperadapter_vl_module1(x, hyper_generator, hw_shape, text_features=text_features)
            elif self.finetune_mode == 'hyperadapter_esod_seeker':
                x = self.hyperadapter_esod_seeker_module1(x, hyper_generator, hw_shape, gt_info=gt_info)
                if isinstance(x, tuple):
                    x, seg_loss_1 = x
            elif self.finetune_mode == 'hyperadapter_multi':
                x = self.hyperadapter_multi_module1(x, hyper_generator, hw_shape)
            elif self.finetune_mode == 'hyperadapter_mona':
                x = self.hyperadapter_mona_module1(x, hyper_generator, hw_shape)
            # elif self.finetune_mode == 'adapter':
            #     x = self.adapter_module1(x) #mona原论文里面的adapter位置和这里不一致

            identity = x
            if self.finetune_mode == 'adapter_former':
                adapt_x = self.adapter_former_module1(x)
            x = self.norm2(x)
            if self.finetune_mode == 'vfmadapter':
                tmpx = self.vfmadapter_query_projector(x)
                q = self.vfmadapter_ffn_queries.expand(tmpx.shape[0], -1, -1)
                sim = torch.matmul(q, tmpx.transpose(1, 2))
                attn = F.softmax(sim, dim=-1)
                z_per_query = torch.matmul(attn, tmpx)
                z = z_per_query.mean(dim=1)
                tmpx =  self.vfmadapter_ffn_adapter(x,z,hyper_generator, hw_shape)
            x = self.ffn(x, identity=identity)
            if self.finetune_mode == 'vfmadapter':
                x = x + tmpx

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
            elif self.finetune_mode == 'query_seeker':
                x = self.query_seeker_module2(x, hw_shape, text_features=text_features)
            elif self.finetune_mode == 'esod_query_seeker':
                x = self.esod_query_seeker_module2(x, hw_shape, text_features=text_features,gt_info=gt_info)
                if isinstance(x, tuple):
                    x, aux_loss_2 = x
            elif self.finetune_mode == 'obj_seeker': #来自ESOD的有gt的模块
                x = self.obj_seeker_module2(x, hw_shape, gt_info=gt_info)
                if isinstance(x, tuple):
                    x, seg_loss_2 = x
            elif self.finetune_mode == 'visual_seeker':
                x = self.visual_seeker_module2(x, hw_shape, gt_info=gt_info)
            elif self.finetune_mode == 'classaware_visual_seeker':
                x = self.classaware_visual_seeker_module2(x, hw_shape, gt_info=gt_info)
            elif self.finetune_mode == 'omni_seeker':
                x = self.omni_seeker_module2(x, hw_shape,text_features=text_features)
            elif self.finetune_mode == 'hyperadapter':
                x = self.hyperadapter_module2(x, hyper_generator, hw_shape)
            elif self.finetune_mode == 'multiscale_hyperadapter':
                x = self.multiscale_hyperadapter_module2(x, hyper_generator, hw_shape)
            elif self.finetune_mode == 'hyperadapter_vl':
                x = self.hyperadapter_vl_module2(x, hyper_generator, hw_shape, text_features=text_features)
            elif self.finetune_mode == 'hyperadapter_esod_seeker':
                x = self.hyperadapter_esod_seeker_module2(x, hyper_generator, hw_shape, gt_info=gt_info)
                if isinstance(x, tuple):
                    x, seg_loss_2 = x
            elif self.finetune_mode == 'hyperadapter_multi':
                x = self.hyperadapter_multi_module2(x, hyper_generator, hw_shape)
            elif self.finetune_mode == 'hyperadapter_mona':
                x = self.hyperadapter_mona_module2(x, hyper_generator, hw_shape)
            # elif self.finetune_mode == 'adapter':
            #     x = self.adapter_module2(x)

            elif self.finetune_mode == 'adapter_former':
                x = x+adapt_x

            if self.text_consistency_loss and self.finetune_mode == 'seeker':
                return (x, consistency_loss_1+consistency_loss_2)
            elif self.text_consistency_loss and self.finetune_mode == 'query_seeker':
                return (x, aux_loss_1+aux_loss_2)
            elif self.text_consistency_loss and self.finetune_mode == 'esod_query_seeker':
                return (x, aux_loss_1+aux_loss_2)
            elif self.esod_loss and gt_info is not None:
                return (x,seg_loss_1+seg_loss_2)
            else:
                return x

        # 当共享的 hyper_generator 参与计算图时，避免使用 reentrant checkpoint 以防 DDP 冲突
        if self.with_cp and x.requires_grad and hyper_generator is None:
            x = cp.checkpoint(_inner_forward, x, text_features, gt_info)
        else:
            x = _inner_forward(x,text_features,gt_info,hyper_generator)

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
        tokens_of_interest (list): 用于obj_seeker的tokens_of_interest. Default: None.
        esod_loss (bool): 是否使用ESOD损失,用于obj_seeker. Default: False.
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
                 esod_loss=False,
                 stage_index=0,
                 adapter_stages=None,
                 ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        self.text_consistency_loss = text_consistency_loss
        self.esod_loss = esod_loss
        self.stage_index = stage_index
        self.adapter_stages = adapter_stages
        #vpt参数
        self.deep_prompt = deep_prompt
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if self.deep_prompt and self.prompt_location != "prepend":
            raise ValueError("deep prompt mode for swin is only applicable to prepend")
        
        for i in range(depth):
            # 决定当前stage是否使用adapter
            use_adapter_in_stage = False
            if adapter_stages is not None:
                # 如果指定了adapter_stages，则只在指定的stage使用adapter
                use_adapter_in_stage = (stage_index in adapter_stages)
            else:
                # 如果没有指定adapter_stages，则根据finetune_mode决定
                use_adapter_in_stage = (finetune_mode is not None)
            
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
                finetune_mode=finetune_mode if use_adapter_in_stage else None,
                num_prompts=num_prompts,
                prompt_location=prompt_location,
                text_dim=text_dim,
                text_consistency_loss=text_consistency_loss,
                tokens_of_interest=tokens_of_interest,
                esod_loss =  esod_loss,
                )
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
    def forward(self, x, hw_shape,text_features=None,deep_prompt=None,gt_info=None, hyper_generator=None):
        if self.deep_prompt:
            assert deep_prompt is not None
            return self.forward_deep(x, hw_shape, text_features,deep_prompt,gt_info)
        if self.text_consistency_loss:
            consistency_loss = torch.zeros(1, device=x.device)
        elif self.esod_loss and gt_info is not None:
            seg_loss = torch.zeros(1, device=x.device)
        for block in self.blocks:
            x = block(x, hw_shape,text_features,gt_info, hyper_generator=hyper_generator)
            if isinstance(x, tuple) and self.text_consistency_loss:
                x, consistency_loss_i = x
                consistency_loss += consistency_loss_i
            elif isinstance(x, tuple) and self.esod_loss and gt_info is not None:
                x,seg_loss_i = x
                seg_loss += seg_loss_i
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            if self.text_consistency_loss:
                return x_down, down_hw_shape, x, hw_shape, consistency_loss
            elif self.esod_loss and gt_info is not None:
                return x_down, down_hw_shape, x, hw_shape, seg_loss
            else:
                return x_down, down_hw_shape, x, hw_shape
        else:
            if self.text_consistency_loss:
                return x, hw_shape, x, hw_shape, consistency_loss
            elif self.esod_loss and gt_info is not None:
                return x, hw_shape, x, hw_shape, seg_loss
            else:
                return x, hw_shape, x, hw_shape

    def forward_deep(self, x, hw_shape,text_features=None,deep_prompt=None,gt_info=None, hyper_generator=None):
        # forwards for deep prompt
        assert self.deep_prompt
        # only support prepend
        assert self.prompt_location == "prepend"
        if self.text_consistency_loss:
            consistency_loss = torch.zeros(1, device=x.device)
        elif self.esod_loss and gt_info is not None:
            seg_loss = torch.zeros(1, device=x.device)
        # add the prompt embed before each blk call
        B = x.shape[0]  # batchsize
        num_blocks = len(self.blocks)
        if deep_prompt.shape[0] != num_blocks:
            # first layer
            #Swin的第一个stage的deep prompt数量是depths[0] - 1，而不是num_blocks（因为第一个block用shallow prompt，后续每个block前插入deep prompt）
            for i in range(num_blocks):
                if i == 0:
                    x = self.blocks[i](x, hw_shape,text_features,gt_info, hyper_generator=hyper_generator)

                else:
                    prompt_emb = deep_prompt[i - 1].expand(B, -1, -1)
                    x = torch.cat(
                        (prompt_emb, x[:, self.num_prompts:, :]),
                        dim=1
                    )
                    x = self.blocks[i](x, hw_shape,text_features,gt_info, hyper_generator=hyper_generator)
                    if isinstance(x, tuple) and self.text_consistency_loss:
                        x, consistency_loss_i = x
                        consistency_loss += consistency_loss_i
                    elif isinstance(x, tuple) and self.esod_loss and gt_info is not None:
                        x,seg_loss_i = x
                        seg_loss += seg_loss_i
        else:
            # other layers
            for i in range(num_blocks):
                prompt_emb = deep_prompt[i].expand(B, -1, -1)
                x = torch.cat(
                    (prompt_emb, x[:, self.num_prompts:, :]),
                    dim=1
                )
                x = self.blocks[i](x, hw_shape,text_features,gt_info, hyper_generator=hyper_generator)
                if isinstance(x, tuple) and self.text_consistency_loss:
                    x, consistency_loss_i = x
                    consistency_loss += consistency_loss_i
                elif isinstance(x, tuple) and self.esod_loss and gt_info is not None:
                    x,seg_loss_i = x
                    seg_loss += seg_loss_i
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            if self.text_consistency_loss:
                return x_down, down_hw_shape, x, hw_shape, consistency_loss
            elif self.esod_loss and gt_info is not None:
                return x_down, down_hw_shape, x, hw_shape, seg_loss
            else:
                return x_down, down_hw_shape, x, hw_shape
        else:
            if self.text_consistency_loss:
                return x, hw_shape, x, hw_shape, consistency_loss
            elif self.esod_loss and gt_info is not None:
                return x, hw_shape, x, hw_shape, seg_loss
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
import numpy as np
## 以下代码来自esod部分的设计，先做一个Objseeker，然后再尝试在这个基础上做一个ObjSeekerEnhancer
class ObjSeeker(nn.Module):
    def __init__(self,
                 in_dim,
                 down_project_dim=64,#降维维度
                 dropout_rate=0.1,
                 ):
        super().__init__()
        
        self.in_dim = in_dim
        self.projected_dim = down_project_dim
        # --- 核心 Adapter 结构 ---
        # 1. 降维
        self.down_project = nn.Linear(in_dim, down_project_dim)
        # 2. 非线性激活与 Dropout (紧跟降维之后)
        self.nonlinear_activation = F.gelu
        self.dropout = nn.Dropout(dropout_rate)

        self.dwconv = DWConvForObjSeeker(down_project_dim, down_project_dim, kernel_size=13, stride=1)
        self.heatmap_conv = Segmenter(nc=1,ch=down_project_dim)
        # 3. 升维
        self.up_project = nn.Linear(down_project_dim, in_dim)
        self.gamma = nn.Parameter(torch.tensor(1e-1))

    def forward(self, x, hw_shapes=None, gt_info=None):
        identity = x
        if gt_info is not None:
            mask,weight = gen_adapter_mask(gt_info,hw_shapes)
        x = self.down_project(x)
        pred = self.nonlinear_activation(x)
        x = self.dropout(pred)
        
        b, n, c = pred.shape
        h, w = hw_shapes
        pred = pred.reshape(b, h, w, c).permute(0, 3, 1, 2)
        # = pred.permute(0, 2, 3, 1).reshape(b, n, c)
        pred = self.dwconv(pred)
        pred = self.heatmap_conv(pred)

        x = self.up_project(x)
        if gt_info is not None:
            loss_seg = compute_loss_seg(pred,mask,weight)
            return identity+x*self.gamma,loss_seg*0.1
        else:
            return identity+x*self.gamma
        #目前根据返回的是不是tuple判断是不是训练状态


##bbox转高斯mask
def gaussian2D(shape, sigma=1., thresh=None):
    #import pdb;pdb.set_trace()
    m, n = [(ss - 1.) / 2. for ss in shape]
    # print(f'm is {m},type is {type(m)}')
    # print(f'n is {n},type is {type(n)}')
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    
    epsilon = 1e-9
    if thresh is None and sigma is not None:
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
    elif thresh is not None and sigma is None:
        thresh += 1e-6
        var1 = n ** 2 / math.log(thresh) + epsilon
        var2 = m ** 2 / math.log(thresh) + epsilon
        h = np.exp(x * x / var1 + y * y / var2)
        h *= 1. / h.max()  # in case even number
        # h += (h - thresh) * (1. - h.max()) / (h.max() - thresh)
        h[h < thresh] = 0.0
    else:
        raise ValueError('invalid gaussian2D parameters')
    
    return h

def gen_adapter_mask(gt_info, hw_shape, thresh=0.5):
    """
    为adapter生成mask，参考get_feature_positions的坐标转换逻辑
    
    Args:
        gt_info: GT信息字典，包含bboxes, labels, num_gt_per_image
        hw_shape: 特征图尺寸 (H, W)
        thresh: 高斯阈值，默认0.5
    
    Returns:
        mask: 生成的mask [H, W]
        weight: 对应的权重 [H, W]
    """
    # 默认启用cls_ratio
    cls_ratio = [1.83, 5.35, 13.82, 1.00, 5.80, 11.25, 30.11, 44.63, 24.45, 4.89]  # train set
    
    H, W = hw_shape
    B = len(gt_info['bboxes'])  # batch size
    
    # 初始化mask和weight [B, H, W]
    mask = np.zeros((B, H, W), dtype=np.float32)
    weight = np.ones((B, H, W), dtype=np.float32)
    
    # 获取GT在特征图上的位置
    feature_positions = get_feature_positions(gt_info, hw_shape)
    # print(f'feature_positions: {feature_positions}')
    # print(f'gt_info: {gt_info}')
    # print(f'hw_shape: {hw_shape}')
    for batch_idx, num_gt in enumerate(gt_info['num_gt_per_image']):
        if num_gt == 0:
            continue
            
        batch_positions = feature_positions[batch_idx]  # [num_gt, 4]
        batch_labels = gt_info['labels'][batch_idx]    # [num_gt]
        
        for gt_idx, (bbox, label) in enumerate(zip(batch_positions, batch_labels)):
            x1, y1, x2, y2 = bbox
            
            # 确保坐标在有效范围内
            x1 = max(0, min(x1, W-1))
            y1 = max(0, min(y1, H-1))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            
            # 检查bbox是否有效
            if x2 <= x1 or y2 <= y1:
                continue
                
            # 计算bbox的宽高
            w, h = x2 - x1, y2 - y1
            #print(f'w: {w}, h: {h}')
            # 生成高斯mask
            gaussian = gaussian2D((int(h), int(w)), sigma=None, thresh=thresh)
            
            # 将高斯mask应用到对应区域 [注意：现在是batch维度]
            masked_hm = mask[batch_idx, y1:y2, x1:x2]
            np.maximum(masked_hm, gaussian, out=masked_hm)
            
            # 计算相对面积比例权重
            area = w * h / (H * W)  # 相对面积
            
            # 小目标和大目标的相对比例设置
            area_min, area_max = 0.01, 0.1  # 相对面积阈值
            
            if area < area_min:
                r_size = (area_min / area) ** 0.5
            elif area > area_max:
                r_size = (area / area_max) ** 0.3
            else:
                r_size = 1.0
            
            # 类别权重
            if label < len(cls_ratio):
                r_cls = cls_ratio[label] ** 0.7
            else:
                r_cls = 1.0
            
            # 综合权重
            r = max(r_size, r_cls)
            
            # 应用权重到对应区域
            masked_wt = weight[batch_idx, y1:y2, x1:x2]
            curr_wt = np.zeros_like(masked_wt) + math.log(r) + 1.
            curr_wt *= (gaussian > 0).astype(mask.dtype)
            np.maximum(masked_wt, curr_wt, out=masked_wt)
    
    # 确保mask值在[0, 1]范围内
    mask = np.clip(mask, 0, 1)
    
    return mask, weight
# 参考ESOD
# def target2mask(targets, shape, nc, stride=8, thresh=0.5):
# #生成mask
# def gen_mask(label_path, image, cls_ratio=False, thresh=0.5, sam_only=False):

def compute_loss_seg(p, masks, weight=None):
    device = p.device
    # 确保masks是tensor且在正确设备上
    if not isinstance(masks, torch.Tensor):
        masks = torch.from_numpy(masks).to(device)
    else:
        masks = masks.to(device)
    
    # 确保weight是tensor且在正确设备上
    if weight is not None:
        if not isinstance(weight, torch.Tensor):
            weight = torch.from_numpy(weight).to(device)
        else:
            weight = weight.to(device)
    
    # 确保数据类型正确
    masks = masks.float()
    if weight is not None:
        weight = weight.float()
     # 确保masks有正确的维度 [B, H, W] -> [B, 1, H, W]
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)  # [B, 1, H, W]
    
    # 确保weight有正确的维度 [B, H, W] -> [B, 1, H, W]
    if weight is not None and weight.dim() == 3:
        weight = weight.unsqueeze(1)  # [B, 1, H, W]

    bs, nc, ny, nx = masks.shape
    assert nc == 1
    lpixl = torch.zeros(1, device=device)
    # weight = None
    lpixl += F.binary_cross_entropy_with_logits(p, masks, weight=weight)


    return lpixl

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class DWConvForObjSeeker(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size=13,stride=1):
        super().__init__()
        self.dwconv = nn.Conv2d(in_dim,out_dim,kernel_size=kernel_size,stride=stride,padding=autopad(kernel_size),groups=math.gcd(in_dim,out_dim))
        self.bn = nn.BatchNorm2d(out_dim)
        self.act = nn.SiLU() 
        
    def forward(self,x):
        return self.act(self.bn(self.dwconv(x)))

class Segmenter(nn.Module):
    def __init__(self, nc=10, ch=64):
        super(Segmenter, self).__init__()
        self.m = nn.Conv2d(ch, nc, 1)  # output conv
    
    def forward(self, x):
        return self.m(x)

from torchvision.ops import roi_align


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
        self.gamma = nn.Parameter(torch.tensor(1.0))



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
            consistency_loss = feature_consistency_loss(activated_features, projected_text_feat_for_loss.detach())
            #原来是：consistency_loss = feature_consistency_loss(activated_features, projected_text_feat_for_loss)
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

class QuerySeekerAdapter(nn.Module):
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
                 ):
        super().__init__()
        
        self.m = m
        self.k = k
        self.in_dim = in_dim
        self.projected_dim = down_project_dim
        self.text_dim = text_dim
        # --- 核心 Adapter 结构 ---
        # 1. 降维
        self.down_project = nn.Linear(in_dim, down_project_dim)
        # 2. 非线性激活与 Dropout (紧跟降维之后)
        self.nonlinear_activation = F.gelu
        self.dropout = nn.Dropout(dropout_rate)
        # 3. 升维
        self.up_project = nn.Linear(down_project_dim, in_dim)
        # --- 结构结束 ---
        # Query 初始化为 nn.Parameter，起到码表的作用
        self.m_queries = nn.Parameter(torch.randn(1, m, down_project_dim))
        self.text_projector = nn.Linear(text_dim,down_project_dim)
        self.text_nonlinear_activation = F.gelu
        self.text_reconstructor = nn.Linear(down_project_dim, text_dim)
        self.image_reconstructor = nn.Linear(down_project_dim, down_project_dim)
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
        #1.常规adapter操作：对视觉特征进行降维
        #2.文本量化投影：将文本投影降维，并通过欧式距离量化到query上
        #3.辅助损失1：query尝试重建到文本原特征，L2Norm计算损失，保证query和文本的关联
        #4.辅助损失2：query尝试重建到视觉池化特征上，同样L2Norm计算损失，保证query和视觉的关联
        #5.辅助损失3：query和文本本身投影特征，L2Norm计算损失，保证query和文本投影的更新
        #6.基于query寻找相似度最高topk的图像特征块
        #7.稀疏attn增强topk特征
        #8.升维+残差
        assert text_features is not None, "text_features 不能为 None"
        B, N, C_in = image_features.shape
        identity = image_features

        # --- 1. 对输入特征应用 Adapter 前半部分：降维 -> GELU -> Dropout ---
        projected_features = self.down_project(image_features)
        activated_features = self.dropout(self.nonlinear_activation(projected_features)) # (B, N, C_proj)

        # --- 2. 文本量化投影：将文本投影降维，并通过欧式距离量化到query上 ---
        # 使用文本投影层将文本特征投影到降维空间
        text_features = text_features.detach()
        projected_text_feat = self.text_projector(text_features)  # (B, L, C_proj)
        projected_text_feat = self.text_nonlinear_activation(projected_text_feat)
        
        # 使用全部文本token进行计算
        L_query = self.m
        L_text = projected_text_feat.shape[1]
        queries = self.m_queries.expand(B, -1, -1)  # (B, m, C_proj)
        norm_queries = F.normalize(queries, p=2, dim=-1)
        norm_text_feat = F.normalize(projected_text_feat, p=2, dim=-1)
        
        # 计算query与文本的相似度
        query_text_similarity = torch.bmm(norm_queries, norm_text_feat.transpose(1, 2))  # (B, m, L)
        
        # 选择最相似的query进行量化 (每个文本token对应一个最相似的query)
        quantized_query_idx = query_text_similarity.argmax(dim=1)  # (B, L)
        #quantized_queries = queries[torch.arange(B).unsqueeze(1), quantized_query_idx]  # (B, L, C_proj)
        quantized_queries = torch.gather(
            queries, dim=1,
            index=quantized_query_idx.unsqueeze(-1).expand(-1, -1, queries.size(-1))
        )  # (B, L, C_proj)
        # --- 3-5. 计算辅助损失 ---
        # 辅助损失1：query尝试重建到文本原特征 (使用全部文本token)
        text_reconstructed = self.text_reconstructor(quantized_queries)  # (B, L, text_dim)
        loss1 = 0.1 * F.mse_loss(text_reconstructed, text_features)
        
        # 辅助损失2：query尝试重建到视觉池化特征
        image_pooled = activated_features.mean(dim=1, keepdim=True)  # (B, 1, C_proj)
        image_pooled_expanded = image_pooled.expand(-1, L_text, -1)  # (B, L, C_proj)
        image_reconstructed = self.image_reconstructor(quantized_queries)  # (B, L, C_proj)
        loss2 = 0.1 * F.mse_loss(image_reconstructed, image_pooled_expanded)
        
        # 辅助损失3：像VQVAE一样，让query特征和文本降维后的特征分别梯度解耦然后算损失
        # 对quantized_queries和projected_text_feat分别进行梯度截断
        quantized_queries_detached = quantized_queries.detach()
        projected_text_feat_detached_for_loss = projected_text_feat.detach()
        loss3 = 0.1 *F.mse_loss(quantized_queries_detached, projected_text_feat)+0.25*0.1*F.mse_loss(quantized_queries, projected_text_feat_detached_for_loss)
        
        # 总辅助损失
        #aux_loss = (loss1 + loss3)*5
        aux_loss = (loss1 + loss2 + loss3)*2

        # --- 6. 基于query寻找相似度最高topk的图像特征块 ---
        # 使用量化后的query计算与图像特征的相似度
        norm_quantized_queries = F.normalize(quantized_queries, p=2, dim=-1)  # (B, L, C_proj)
        norm_image_features = F.normalize(activated_features, p=2, dim=-1)  # (B, N, C_proj)
        image_scores = torch.bmm(norm_image_features, norm_quantized_queries.transpose(1, 2))  # (B, N, L)
        # 在文本token维度取最大值，得到每个图像token与所有文本token的最大相似度
        image_scores, _ = image_scores.max(dim=2)  # (B, N)
        
        # 使用STE生成topk mask
        mask = ste_topk_mask(image_scores, self.k)  # (B, N)
        topk_indices = mask.nonzero(as_tuple=False).view(B, self.k, 2)[:,:,1]  # (B, k)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.projected_dim)
        sparse_image_tokens = torch.gather(activated_features, 1, topk_indices_expanded)

        # --- 7. 稀疏attn增强topk特征 ---
        # 将量化后的query与稀疏图像token结合
        combined_sequence = torch.cat([quantized_queries, sparse_image_tokens], dim=1)  # (B, L+k, C_proj)
        
        # 交互: Norm -> Self-Attention -> Add
        attention_input = self.norm(combined_sequence)
        attention_output, _ = self.interaction_attention(
            query=attention_input, key=attention_input, value=attention_input
        )
        enhanced_sequence = combined_sequence + attention_output
        
        # 分离出增强后的图像token
        enhanced_sparse_tokens = enhanced_sequence[:, L_text:, :]  # (B, k, C_proj)

        # --- 8. 升维+残差 ---
        # 创建一个与激活后特征图形状相同的零张量
        delta_x_projected = torch.zeros_like(activated_features)
        delta_x_projected.scatter_(1, topk_indices_expanded, enhanced_sparse_tokens)
        
        # 应用 Adapter 后半部分：升维
        delta_x = self.up_project(delta_x_projected)  # -> (B, N, C_in)
        
        return identity + delta_x * self.gamma, aux_loss

    def old_forward(self, image_features, hw_shapes=None, text_features=None):
        #1.常规adapter操作：对视觉特征进行降维
        #2.文本投影：将文本投影降维或者截断，降维则产生和视觉池化特征匹配的损失
        #3.文本寻找相似度最高topk
        #4.可学习query稀疏attn增强topk特征，ema更新query
        #5.升维+残差
        # return identity + delta_x * self.gamma
        return image_features



class HyperNetworksGenerator(nn.Module):

    def __init__(self, n_z: int, n_in: int, n_out: int, f_size: int = 3, init_scale: float = 1e-2):
        super(HyperNetworksGenerator, self).__init__()

        # 参数顺序与 HyperNetworksGenerator 保持一致
        self.n_z = n_z
        d = n_in*n_z
        self.d = d
        self.n_in = n_in
        self.n_out = n_out
        self.f_size = f_size
        self.init_scale = init_scale

        output_dim = n_in * f_size * f_size * n_out

        # 手动参数（不使用 nn.Linear），但使用现代初始化方式
        self.w2 = nn.Parameter(torch.empty(n_z, d))      # 对应第一层权重: z -> d
        self.b2 = nn.Parameter(torch.empty(d))           # 第一层偏置
        self.w1 = nn.Parameter(torch.empty(d, output_dim))  # 对应第二层权重: d -> 输出
        self.b1 = nn.Parameter(torch.empty(output_dim))     # 第二层偏置

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Safer initialization for dynamic conv generator
        nn.init.kaiming_normal_(self.w2, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.b2)
        nn.init.kaiming_normal_(self.w1, mode='fan_in', nonlinearity='linear')
        # Scale down final projection to keep initial dynamic kernels small
        with torch.no_grad():
            self.w1.mul_(self.init_scale)
        nn.init.zeros_(self.b1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 确保 z 与模型参数在同一设备上
        z = z.to(next(self.parameters()).device)
        
        # 支持 [n_z] 或 [batch, n_z]
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # 第一层: z -> d
        hidden = torch.matmul(z, self.w2) + self.b2  # [batch, d]
        # 第二层: d -> n_in * f * f * n_out
        weights_flat = torch.matmul(hidden, self.w1) + self.b1  # [batch, output_dim]

        # 还原为卷积核形状，与 HyperNetworksGenerator 一致
        K = weights_flat.reshape(-1, self.n_in, self.f_size, self.f_size, self.n_out)
        # Optional: clip/tanh to improve early stability
        K = torch.tanh(K)

        if squeeze_output:
            K = K.squeeze(0)

        return K

class HyperConv2d(nn.Module):
    """
    使用 HyperNetworks 生成的卷积层
    """
    
    def __init__(self, 
                 stride: int = 1,
                 padding: int = 1):
        super(HyperConv2d, self).__init__()
        
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor, z: torch.Tensor, generator: "HyperNetworksGenerator") -> torch.Tensor:
        # 生成带 batch 的权重: [B, n_in, k, k, n_out]
        weights = generator(z)
        
        # 确保权重与输入在同一设备上
        weights = weights.to(x.device)

        # 输入: [B, C_in, H, W]
        B, C_in, H, W = x.shape
        assert weights.dim() == 5, "generator(z) 应返回 [B, n_in, k, k, n_out]"

        # 重排为 [B, C_out, C_in, k, k]
        weights = weights.permute(0, 4, 1, 2, 3)
        B_w, C_out, C_in_w, K, _ = weights.shape

        assert B_w == B, "输入与生成权重的 batch 数不匹配"

        # 使用 grouped conv 实现 per-sample 卷积
        x_group = x.reshape(1, B * C_in, H, W)
        # Fan-in scaling to prevent large activations
        scale = 1.0 / math.sqrt(C_in * K * K)
        w_group = (weights * scale).reshape(B * C_out, C_in_w, K, K)
        y = F.conv2d(x_group, w_group, stride=self.stride, padding=self.padding, groups=B)
        y = y.reshape(B, C_out, y.shape[-2], y.shape[-1])
        return y
#TODO:手动复现版本
class HyperAdapter(BaseModule):
    def __init__(self,
                 in_dim,
                 m=16):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        # 不将generator作为模块参数存储，而是通过外部传入
        self.m_queries = nn.Parameter(torch.randn(1, m, 64))
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        
        # 预创建HyperConv2d，避免每次forward都创建新实例
        self.hyper_conv = None

    def forward(self, x, generator, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        # 计算 z：基于 m_queries 与 project1 的相似度得到注意力分数，对所有特征加权平均
        # project1: [b, c, h, w] -> [b, hw, c]
        x_flat = project1.reshape(b, c, h * w).permute(0, 2, 1)
        # m_queries: [1, m, c] -> [b, m, c]
        q = self.m_queries.expand(b, -1, -1)
        # 相似度: [b, m, c] x [b, c, hw] -> [b, m, hw]
        sim = torch.matmul(q, x_flat.transpose(1, 2))
        attn = F.softmax(sim, dim=-1)
        # 加权求和: [b, m, hw] x [b, hw, c] -> [b, m, c]
        z_per_query = torch.matmul(attn, x_flat)
        # 对 m 维求平均，得到 [b, c]
        z = z_per_query.mean(dim=1)
        
        # 使用预创建的HyperConv2d，避免每次forward都创建新实例
        if self.hyper_conv is None:
            self.hyper_conv = HyperConv2d(padding=generator.f_size//2)
        project1 = self.hyper_conv(project1, z, generator)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2

class VFMAdapter(BaseModule):
    def __init__(self,
                 in_dim):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.relu
        self.project2 = nn.Linear(64, in_dim)
        # 不将generator作为模块参数存储，而是通过外部传入
        #self.m_queries = nn.Parameter(torch.randn(1, m, 64))
        # 预创建HyperConv2d，避免每次forward都创建新实例
        self.hyper_conv = None

    def forward(self, x, z, generator, hw_shapes=None):
        #输入z是聚合向量，相比于hyperadapter，使用的z输入维度更大
        #identity = x
        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        # print("*"*100)
        # print("already in vfm adapter,before hyper conv")
        # 使用预创建的HyperConv2d，避免每次forward都创建新实例
        if self.hyper_conv is None:
            self.hyper_conv = HyperConv2d(padding=generator.f_size//2)
        project1 = self.hyper_conv(project1, z, generator)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        # print("!"*100)
        # print("already in vfm adapter,after hyper conv")
        nonlinear = self.nonlinear(project1)
        project2 = self.project2(nonlinear)

        return project2

class MultiscaleDepthwiseSeparableHypernetworksGenerator(nn.Module):
    """
    生成多尺度深度可分离卷积核的超网络
    
    设计特点：
    1. 生成3路多尺度卷积核：1x1, 3x3, 5x5 深度可分离卷积
    2. 每路都是标准的深度可分离卷积：depthwise + pointwise
    3. 三路结果进行平均融合，而不是拼接
    4. 相比于标准卷积，参数数量大幅减少
    
    公式：
    - Depthwise separable conv = Depthwise conv + Pointwise conv
    - 参数数量: K*K*C_in (depthwise) + C_in*C_out (pointwise) << K*K*C_in*C_out
    """
    
    def __init__(self, 
                 n_z: int,           # 隐变量维度
                 n_in: int,          # 输入通道数
                 n_out: int,         # 输出通道数
                 init_scale: float = 1e-2):
        super(MultiscaleDepthwiseSeparableHypernetworksGenerator, self).__init__()
        
        self.n_z = n_z
        self.n_in = n_in
        self.n_out = n_out
        self.init_scale = init_scale
        
        d = n_in * n_z  # 第一层隐层维度
        self.d = d
        
        # 多尺度深度可分离卷积的总参数数量
        # 1x1: n_in * n_out (pointwise only)
        # 3x3: 9 * n_in (depthwise) + n_in * n_out (pointwise)
        # 5x5: 25 * n_in (depthwise) + n_in * n_out (pointwise)
        output_dim = (n_in * n_out +                    # 1x1 pointwise weights
                     (9 * n_in + n_in * n_out) +       # 3x3 depthwise + pointwise
                     (25 * n_in + n_in * n_out))       # 5x5 depthwise + pointwise
        
        # 超网络参数（两层MLP）
        self.w2 = nn.Parameter(torch.empty(n_z, d))
        self.b2 = nn.Parameter(torch.empty(d))
        self.w1 = nn.Parameter(torch.empty(d, output_dim))
        self.b1 = nn.Parameter(torch.empty(output_dim))
        
        self._init_parameters()
    
    def _init_parameters(self) -> None:
        """初始化超网络参数"""
        nn.init.kaiming_normal_(self.w2, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.b2)
        nn.init.kaiming_normal_(self.w1, mode='fan_in', nonlinearity='linear')
        with torch.no_grad():
            self.w1.mul_(self.init_scale)
        nn.init.zeros_(self.b1)
    
    def forward(self, z: torch.Tensor) -> dict:
        """
        生成多尺度深度可分离卷积的核参数
        
        Args:
            z: [B, n_z] 或 [n_z] 的隐变量
        
        Returns:
            dict 包含:
                'pw_1x1': [B, n_out, n_in, 1, 1] - 1x1 pointwise weights
                'dw_3x3': [B, n_in, 1, 3, 3] - 3x3 depthwise weights
                'pw_3x3': [B, n_out, n_in, 1, 1] - 3x3 pointwise weights
                'dw_5x5': [B, n_in, 1, 5, 5] - 5x5 depthwise weights
                'pw_5x5': [B, n_out, n_in, 1, 1] - 5x5 pointwise weights
        """
        z = z.to(next(self.parameters()).device)
        
        # 处理输入维度
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = z.shape[0]
        
        # 两层MLP生成参数
        hidden = torch.matmul(z, self.w2) + self.b2  # [B, d]
        weights_flat = torch.matmul(hidden, self.w1) + self.b1  # [B, output_dim]
        weights_flat = torch.tanh(weights_flat)  # 稳定性
        
        # 解析生成的权重
        offset = 0
        
        # 1x1 pointwise: [B, n_out, n_in, 1, 1]
        pw_1x1_size = self.n_in * self.n_out
        pw_1x1 = weights_flat[:, offset:offset+pw_1x1_size]
        pw_1x1 = pw_1x1.reshape(batch_size, self.n_out, self.n_in, 1, 1)
        offset += pw_1x1_size
        
        # 3x3 depthwise: [B, n_in, 1, 3, 3]
        dw_3x3_size = 9 * self.n_in
        dw_3x3 = weights_flat[:, offset:offset+dw_3x3_size]
        dw_3x3 = dw_3x3.reshape(batch_size, self.n_in, 1, 3, 3)
        offset += dw_3x3_size
        
        # 3x3 pointwise: [B, n_out, n_in, 1, 1]
        pw_3x3_size = self.n_in * self.n_out
        pw_3x3 = weights_flat[:, offset:offset+pw_3x3_size]
        pw_3x3 = pw_3x3.reshape(batch_size, self.n_out, self.n_in, 1, 1)
        offset += pw_3x3_size
        
        # 5x5 depthwise: [B, n_in, 1, 5, 5]
        dw_5x5_size = 25 * self.n_in
        dw_5x5 = weights_flat[:, offset:offset+dw_5x5_size]
        dw_5x5 = dw_5x5.reshape(batch_size, self.n_in, 1, 5, 5)
        offset += dw_5x5_size
        
        # 5x5 pointwise: [B, n_out, n_in, 1, 1]
        pw_5x5_size = self.n_in * self.n_out
        pw_5x5 = weights_flat[:, offset:offset+pw_5x5_size]
        pw_5x5 = pw_5x5.reshape(batch_size, self.n_out, self.n_in, 1, 1)
        
        result = {
            'pw_1x1': pw_1x1,  # [B, n_out, n_in, 1, 1]
            'dw_3x3': dw_3x3,  # [B, n_in, 1, 3, 3]
            'pw_3x3': pw_3x3,  # [B, n_out, n_in, 1, 1]
            'dw_5x5': dw_5x5,  # [B, n_in, 1, 5, 5]
            'pw_5x5': pw_5x5,  # [B, n_out, n_in, 1, 1]
        }
        
        if squeeze_output:
            result = {k: v.squeeze(0) for k, v in result.items()}
        
        return result


class MultiscaleDepthwiseSeparableHyperConv2d(nn.Module):
    """
    使用 MultiscaleDepthwiseSeparableHypernetworksGenerator 生成的多尺度卷积层
    
    实现3路多尺度深度可分离卷积：
    - 路径1: 1x1 pointwise卷积
    - 路径2: 3x3 depthwise + 1x1 pointwise
    - 路径3: 5x5 depthwise + 1x1 pointwise
    
    三路结果进行平均融合
    """
    
    def __init__(self, stride: int = 1, padding: int = 1):
        super(MultiscaleDepthwiseSeparableHyperConv2d, self).__init__()
        # 确保stride是整数
        if isinstance(stride, (list, tuple)):
            self.stride = stride[0] if len(stride) > 0 else 1
        else:
            self.stride = stride
        self.padding = padding
    
    def forward(self, x: torch.Tensor, 
                z: torch.Tensor, 
                generator: "MultiscaleDepthwiseSeparableHypernetworksGenerator") -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W] 输入特征
            z: [B, n_z] 或 [n_z] 隐变量
            generator: MultiscaleDepthwiseSeparableHypernetworksGenerator 实例
        
        Returns:
            y: [B, C_out, H, W] 输出特征（三路平均融合）
        """
        # 生成所有路径的权重
        weights = generator(z)  # dict of tensors
        
        B, C_in, H, W = x.shape
        C_out = weights['pw_1x1'].shape[1]  # 从权重形状获取输出通道数
        
        # 确保权重与输入在同一设备上
        device = x.device
        for key in weights:
            if weights[key].device != device:
                weights[key] = weights[key].to(device)
        
        outputs = []
        
        # 路径1: 1x1 pointwise卷积
        pw_1x1 = weights['pw_1x1']  # [B, C_out, C_in, 1, 1]
        scale = 1.0 / math.sqrt(C_in)
        pw_1x1_scaled = (pw_1x1 * scale).squeeze(0)  # [C_out, C_in, 1, 1]
        y1 = F.conv2d(x, pw_1x1_scaled, stride=self.stride, padding=0)
        outputs.append(y1)
        
        # 路径2: 3x3 depthwise + pointwise
        dw_3x3 = weights['dw_3x3']  # [B, C_in, 1, 3, 3]
        pw_3x3 = weights['pw_3x3']  # [B, C_out, C_in, 1, 1]
        
        # Depthwise 3x3
        scale = 1.0 / math.sqrt(9)
        dw_3x3_scaled = (dw_3x3 * scale).squeeze(0)  # [C_in, 1, 3, 3]
        y_dw = F.conv2d(x, dw_3x3_scaled, stride=self.stride, 
                       padding=self.padding, groups=C_in)
        
        # Pointwise 1x1
        scale = 1.0 / math.sqrt(C_in)
        pw_3x3_scaled = (pw_3x3 * scale).squeeze(0)  # [C_out, C_in, 1, 1]
        y2 = F.conv2d(y_dw, pw_3x3_scaled)
        outputs.append(y2)
        
        # 路径3: 5x5 depthwise + pointwise
        dw_5x5 = weights['dw_5x5']  # [B, C_in, 1, 5, 5]
        pw_5x5 = weights['pw_5x5']  # [B, C_out, C_in, 1, 1]
        
        # Depthwise 5x5
        scale = 1.0 / math.sqrt(25)
        dw_5x5_scaled = (dw_5x5 * scale).squeeze(0)  # [C_in, 1, 5, 5]
        y_dw = F.conv2d(x, dw_5x5_scaled, stride=self.stride, 
                       padding=2, groups=C_in)
        
        # Pointwise 1x1
        scale = 1.0 / math.sqrt(C_in)
        pw_5x5_scaled = (pw_5x5 * scale).squeeze(0)  # [C_out, C_in, 1, 1]
        y3 = F.conv2d(y_dw, pw_5x5_scaled)
        outputs.append(y3)
        
        # 三路结果平均融合
        output = torch.stack(outputs, dim=0).mean(dim=0)  # [B, C_out, H, W]
        return output


class MultiscaleHyperAdapter(BaseModule):
    """
    基于多尺度深度可分离卷积超网络的适配器模块
    
    结合了多尺度设计和HyperNetwork的动态生成能力
    """
    
    def __init__(self,
                 in_dim: int,
                 m: int = 16,
                 intermediate_dim: int = 64):
        """
        Args:
            in_dim: 输入维度
            m: 多路查询的数量
            intermediate_dim: 中间投影维度
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.intermediate_dim = intermediate_dim
        
        # 投影层
        self.project1 = nn.Linear(in_dim, intermediate_dim)
        self.nonlinear = F.gelu
        self.dropout = nn.Dropout(p=0.1)
        
        # 输出投影（从多尺度融合回原维度）
        self.project2 = nn.Linear(intermediate_dim, in_dim)
        
        # 层归一化和缩放
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        
        # 多路查询用于生成隐变量z
        self.m_queries = nn.Parameter(torch.randn(1, m, intermediate_dim))
        
        # 预创建卷积层
        #self.hyper_conv = MultiscaleDepthwiseSeparableHyperConv2d(stride=1, padding=1)
        self.hyper_conv = None
    
    def forward(self, x: torch.Tensor, 
                generator: "MultiscaleDepthwiseSeparableHypernetworksGenerator",
                hw_shapes: tuple) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] 输入特征 (N=H*W)
            generator: MultiscaleDepthwiseSeparableHypernetworksGenerator 实例
            hw_shapes: (H, W) 特征图大小
        
        Returns:
            y: [B, N, C] 输出特征
        """
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        
        # 第一层投影
        project1 = self.project1(x)  # [B, N, intermediate_dim]
        
        b, n, c = project1.shape
        h, w = hw_shapes
        
        # 重塑为空间形式
        project1_spatial = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 计算z：基于多路查询与特征的相似度
        x_flat = project1_spatial.reshape(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]
        q = self.m_queries.expand(b, -1, -1)  # [B, m, C]
        
        # 计算注意力分数
        sim = torch.matmul(q, x_flat.transpose(1, 2))  # [B, m, HW]
        attn = F.softmax(sim, dim=-1)
        
        # 加权求和得到每个查询的加权特征
        z_per_query = torch.matmul(attn, x_flat)  # [B, m, C]
        
        # 对所有查询取平均得到最终的z
        z = z_per_query.mean(dim=1)  # [B, C]
        
        # 应用多尺度深度可分离卷积
        if self.hyper_conv is None:
            self.hyper_conv = MultiscaleDepthwiseSeparableHyperConv2d(stride=1, padding=1)
        project1_out = self.hyper_conv(project1_spatial, z, generator)  # [B, C, H, W]
        
        # 重塑回序列形式
        project1_out = project1_out.permute(0, 2, 3, 1).reshape(b, n, c)
        
        # 非线性激活和dropout
        nonlinear = self.nonlinear(project1_out)
        nonlinear = self.dropout(nonlinear)
        
        # 投影回原维度
        project2 = self.project2(nonlinear)  # [B, N, in_dim]
        
        return identity + project2

class MultiscaleHyperAdapterWithNoQueries(BaseModule):
    """
    基于多尺度深度可分离卷积超网络的适配器模块,消除了query以进行消融实验,在multiscale_adapter的基础上进行修改
    
    结合了多尺度设计和HyperNetwork的动态生成能力
    """
    
    def __init__(self,
                 in_dim: int,
                 intermediate_dim: int = 64):
        """
        Args:
            in_dim: 输入维度
            intermediate_dim: 中间投影维度
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.intermediate_dim = intermediate_dim
        
        # 投影层
        self.project1 = nn.Linear(in_dim, intermediate_dim)
        self.nonlinear = F.gelu
        self.dropout = nn.Dropout(p=0.1)
        
        # 输出投影（从多尺度融合回原维度）
        self.project2 = nn.Linear(intermediate_dim, in_dim)
        
        # 层归一化和缩放
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
            
        # 预创建卷积层
        #self.hyper_conv = MultiscaleDepthwiseSeparableHyperConv2d(stride=1, padding=1)
        self.hyper_conv = None
    
    def forward(self, x: torch.Tensor, 
                generator: "MultiscaleDepthwiseSeparableHypernetworksGenerator",
                hw_shapes: tuple) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] 输入特征 (N=H*W)
            generator: MultiscaleDepthwiseSeparableHypernetworksGenerator 实例
            hw_shapes: (H, W) 特征图大小
        
        Returns:
            y: [B, N, C] 输出特征
        """
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        
        # 第一层投影
        project1 = self.project1(x)  # [B, N, intermediate_dim]
        
        b, n, c = project1.shape
        h, w = hw_shapes
        
        # 重塑为空间形式
        project1_spatial = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        z = project1.mean(dim=1)  # [B, C]
        
        # 应用多尺度深度可分离卷积
        if self.hyper_conv is None:
            self.hyper_conv = MultiscaleDepthwiseSeparableHyperConv2d(stride=1, padding=1)
        project1_out = self.hyper_conv(project1_spatial, z, generator)  # [B, C, H, W]
        
        # 重塑回序列形式
        project1_out = project1_out.permute(0, 2, 3, 1).reshape(b, n, c)
        
        # 非线性激活和dropout
        nonlinear = self.nonlinear(project1_out)
        nonlinear = self.dropout(nonlinear)
        
        # 投影回原维度
        project2 = self.project2(nonlinear)  # [B, N, in_dim]
        
        return identity + project2


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
        esod_loss (bool): 是否使用ESOD损失,用于obj_seeker
        adapter_stages (list or None): 指定哪些stage启用adapter。例如[2, 3]表示只在第3和第4个stage启用adapter。
            如果为None，则根据finetune_mode决定是否在所有stage启用adapter。默认: None
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
                 esod_loss=False,
                 adapter_stages=None,
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
        self.esod_loss = esod_loss
        self.adapter_stages = adapter_stages

        if finetune_mode is not None and 'hyperadapter' in finetune_mode and 'hyperadapter_multi' not in finetune_mode and 'multiscale_hyperadapter' not in finetune_mode:
            self.hyper_generator = HyperNetworksGenerator(n_z=64, n_in=64, n_out=64, f_size=3)
        elif finetune_mode is not None and 'hyperadapter_multi' in finetune_mode:
            self.hyper_generator = []
            for i in [1,3,3]:
                self.hyper_generator.append(HyperNetworksGenerator(n_z=64, n_in=64, n_out=64, f_size=i))
        elif finetune_mode is not None and 'multiscale_hyperadapter' in finetune_mode:
            self.hyper_generator = MultiscaleDepthwiseSeparableHypernetworksGenerator(n_z=64, n_in=64, n_out=64)
        elif finetune_mode is not None and 'vfmadapter' in finetune_mode:
            self.hyper_generator = HyperNetworksGenerator(n_z=64, n_in=64, n_out=64, f_size=3)
            # self.hyper_generator = []
            # for i in range(2): # vfm attn和ffn各有其生成器
            #     self.hyper_generator.append(HyperNetworksGenerator(n_z=96, n_in=64, n_out=64, f_size=3))
        else:
            self.hyper_generator = None
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
                esod_loss = esod_loss,
                stage_index=i,
                adapter_stages=self.adapter_stages,
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
            elif finetune_mode == 'query_seeker':
                for name, param in self.named_parameters():
                    if 'query_seeker_module' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'esod_query_seeker':
                for name, param in self.named_parameters():
                    if 'esod_query_seeker_module' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'obj_seeker': #来自ESOD的有gt的模块
                for name, param in self.named_parameters():
                    if 'obj_seeker_module' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'visual_seeker':
                for name, param in self.named_parameters():
                    if 'visual_seeker_module' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'classaware_visual_seeker':
                for name, param in self.named_parameters():
                    if 'classaware_visual_seeker_module' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'omni_seeker':
                for name, param in self.named_parameters():
                    if 'omni_seeker_module' not in name:
                        param.requires_grad = requires_grad
            elif 'hyperadapter' in finetune_mode:
                for name, param in self.named_parameters():
                    if 'hyperadapter' not in name and 'hyper_generator' not in name:
                        param.requires_grad = requires_grad
            elif finetune_mode == 'vfmadapter':
                for name, param in self.named_parameters():
                    if 'vfmadapter' not in name and 'hyper_generator' not in name:
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
        elif self.esod_loss and gt_info is not None:
            seg_loss = torch.zeros(1, device=x.device)
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
                        x, hw_shape, out, out_hw_shape, consistency_loss_i = stage(x, hw_shape,text_features,gt_info=gt_info, hyper_generator=self.hyper_generator)
                        consistency_loss += consistency_loss_i
                    elif self.esod_loss and gt_info is not None:
                        x, hw_shape, out, out_hw_shape, seg_loss_i = stage(x, hw_shape,text_features,gt_info=gt_info, hyper_generator=self.hyper_generator)
                        seg_loss += seg_loss_i
                    else:
                        x, hw_shape, out, out_hw_shape = stage(x, hw_shape,text_features,gt_info=gt_info, hyper_generator=self.hyper_generator)
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
                        x, hw_shape, out, out_hw_shape, consistency_loss_i = stage(x, hw_shape,text_features,deep_prompt_emb,gt_info=gt_info, hyper_generator=self.hyper_generator)
                        consistency_loss += consistency_loss_i
                    elif self.esod_loss and gt_info is not None:
                        x, hw_shape, out, out_hw_shape, seg_loss_i = stage(x, hw_shape,text_features,gt_info=gt_info, hyper_generator=self.hyper_generator)
                        seg_loss += seg_loss_i
                    else:
                        x, hw_shape, out, out_hw_shape = stage(x, hw_shape,text_features,deep_prompt_emb,gt_info=gt_info, hyper_generator=self.hyper_generator)
                    if i in self.out_indices:
                        norm_layer = getattr(self, f'norm{i}')
                        out = norm_layer(out)
                        # remove prompts
                        out = out[:, self.prompt_num_tokens:, :]
                        out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                        outs.append(out)
            if self.text_consistency_loss:
                return outs, consistency_loss
            elif self.esod_loss and gt_info is not None:
                return outs,seg_loss
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
                    x, hw_shape, out, out_hw_shape, consistency_loss_i = stage(x, hw_shape,text_features,gt_info=gt_info, hyper_generator=self.hyper_generator)
                    consistency_loss += consistency_loss_i
                elif self.esod_loss and gt_info is not None:
                    x, hw_shape, out, out_hw_shape, seg_loss_i = stage(x, hw_shape,text_features,gt_info=gt_info, hyper_generator=self.hyper_generator)
                    seg_loss += seg_loss_i
                else:
                    x, hw_shape, out, out_hw_shape = stage(x, hw_shape,text_features,gt_info=gt_info, hyper_generator=self.hyper_generator)
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    out = norm_layer(out)
                    out = out.view(-1, *out_hw_shape,
                                   self.num_features[i]).permute(0, 3, 1,
                                                                 2).contiguous()
                    outs.append(out)
            if self.text_consistency_loss:
                return outs, consistency_loss
            elif self.esod_loss and gt_info is not None:
                return outs,seg_loss
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


