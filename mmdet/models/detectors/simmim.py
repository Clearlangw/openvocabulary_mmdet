import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS
import copy
import warnings
from typing import List, Tuple, Union,Dict

from mmdet.registry import MODELS
from .base import BaseDetector
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig,ConfigType
from ..utils import samplelist_boxtype2tensor

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

@MODELS.register_module()
class SimMIM(BaseDetector):
    """
    SimMIM 模型，集成编码器（SwinTransformerForSimMIM）和解码器，
    用于自监督图像重建任务。解码器部分使用 1×1 卷积转换通道，
    
    Args:
        backbone (dict): 编码器配置字典，传入 SwinTransformerForSimMIM 配置。
    """
    def __init__(self,
                 backbone,
                 patch_size=4,
                 mask_ratio=0.6,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None
                 ):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # 使用 mmdet 注册机制构造编码器
        self.encoder = MODELS.build(backbone)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        # 新的解码器设计：1×1 卷积转换通道至 3（RGB），再使用双线性插值上采样
        self.decoder_conv = nn.Conv2d(
            in_channels=self.encoder.num_features[-1],
            out_channels=3,
            kernel_size=1)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        return False

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return False

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return False

    @property
    def with_shared_head(self) -> bool:
        """bool: whether the detector has a shared head in the RoI Head"""
        return False

    @property
    def with_bbox(self) -> bool:
        """bool: whether the detector has a bbox head"""
        return False

    @property
    def with_mask(self) -> bool:
        """bool: whether the detector has a mask head"""
        return False

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def generate_mask(self, B, h, w, patch_size, mask_ratio,device):
        """
        为每个批次生成掩码，遮挡一定比例的 patch。
        
        Args:
            B (int): 批次大小
            h (int): 输入图像的高度
            w (int): 输入图像的宽度
            patch_size (int): 每个 patch 的大小
            mask_ratio (float): 遮挡比例

        Returns:
            Tensor: 形状为 (B, H_enc, W_enc) 的掩码张量
        """
        # 计算编码器输出的尺寸
        H_enc = (h + patch_size * 8 - 1) // (patch_size * 8)  # ceil(h / (patch_size * 8))
        W_enc = (w + patch_size * 8 - 1) // (patch_size * 8)  # ceil(w / (patch_size * 8))

        # 随机生成掩码索引
        num_patches = H_enc * W_enc
        num_masked_patches = int(num_patches * mask_ratio)
        mask_idx = torch.rand((B, num_patches),device=device).argsort(dim=-1)[:, :num_masked_patches]
        mask = torch.ones((1, H_enc, W_enc),device=device)  # 先生成一个批次的掩码
        # 创建掩码
        for idx in mask_idx:
            row = idx // W_enc
            col = idx % W_enc
            mask[0, row, col] = 0  # 标记该patch位置为掩码

        # 将掩码扩展到整个批次
        mask = mask.expand(B, -1, -1)  # 扩展为 (B, H_enc, W_enc)

        return mask
    
    def expand_mask(self, mask,h,w,patch_size):
        """
        根据编码器输出尺寸，将掩码扩展到patch_embed的size，用于输入到encoder部分。
        Args:
            mask (Tensor): 掩码，形状为 (B, H_enc, W_enc)
        Returns:
            Tensor: 扩展后的掩码，形状为 (B, H, W)
        """
        B,H_enc,W_enc = mask.shape  # 批次大小，直接从掩码形状获取
        H_exp = (h + patch_size - 1) // (patch_size)  # ceil(h / (patch_size))
        W_exp = (w + patch_size  - 1) // (patch_size)  # ceil(w / (patch_size))
       
        mask_expanded = torch.ones((B, H_exp, W_exp), device=mask.device)  # 创建一个全零的掩码，尺寸为原图像尺寸

        for b in range(B):
            for i in range(H_enc):
                for j in range(W_enc):
                    if mask[b, i, j] == 0:  # 如果该位置是掩码
                        # 在 z 的输出尺寸中，扩展为 8x8 区域
                        for h_it in range(i*8,min((i+1)*8,H_exp)):
                            for w_it in range(j*8,min((j+1)*8,W_exp)):
                                mask_expanded[b, h_it, w_it] = 0

        return mask_expanded


    def loss(self, inputs,data_samples):
        """
        Args:
            x=inputs (Tensor): 输入图像，形状 (B, 3, H, W)，例如 H=1400, W=1600
        Returns:
            masked_l1_loss (Tensor): 仅考虑有效部分的 L1 损失
        """
        # 1.计算shape并生成最小的mask(针对z)
        x = inputs 
        losses = dict()
        # print(x[0][0])
        # import sys
        # sys.exit()
        B, _, h, w = x.shape
        mask = self.generate_mask(B,h,w,patch_size=self.patch_size,mask_ratio=self.mask_ratio,device=x.device)
        mask_expanded = self.expand_mask(mask,h,w,patch_size=self.patch_size)
        # x->(h,w) patch_embed->(ceil(h/patch_size),ceil(w/patch_size)) 
        # z ->(ceil(h/patch_size*8),ceil(w/patch_size*8)) 

        # 2.扩展mask并传入encoder中
        z = self.encoder(x,mask_expanded)  
        # print('z.shape is ',z.shape) # z.shape is  torch.Size([2, 768, 23, 31])
        # print('hw_shape is ',hw_shape) # hw_shape is  (23, 31)
        # print("x.shape is ",x.shape)
        # print("mask_idx is ",mask_idx)
        # import sys
        # sys.exit()
        B, C, H_enc,W_enc = z.shape

        # 3. 通过 1×1 卷积转换为 3 通道（RGB）
        x_rec = self.decoder_conv(z)
        
        # 4.归一化恢复图像（标准化）
        x_rec_norm = (x_rec - x_rec.mean()) / (x_rec.std() + 1e-5)  # 使用标准化

        # 5.通过池化将原图像尺寸缩小，作为目标图像
        target_size = (H_enc, W_enc)  # 假设池化后的目标大小为 encoder 输出的尺寸
        x_pooled = F.adaptive_avg_pool2d(x, target_size)  # 使用池化获得目标图像

        # 计算 L1 损失时，只考虑非掩码部分
        l1_loss = torch.abs(x_rec_norm  - x_pooled)  # 计算恢复图像与池化后目标图像之间的 L1 损失
        masked_l1_loss = l1_loss * mask.unsqueeze(1).expand(-1,3,-1,-1)
        # 仅考虑非掩码部分的损失
        masked_l1_loss = masked_l1_loss.sum() / (mask.sum()+1e-6)  # 计算非掩码部分的平均损失
        losses['masked_l1_loss']=masked_l1_loss
        return losses

    def predict(self, inputs,data_samples):
        """
        Args:
            x=inputs (Tensor): 输入图像，形状 (B, 3, H, W)，例如 H=1400, W=1600
        Returns:
        """
        # 1.计算shape并生成最小的mask(针对z)
        x = inputs 
        # print(x[0][0])
        # import sys
        # sys.exit()
        B, _, h, w = x.shape
        mask = self.generate_mask(B,h,w,patch_size=self.patch_size,mask_ratio=self.mask_ratio,device=x.device)
        mask_expanded = self.expand_mask(mask,h,w,patch_size=self.patch_size)
        # x->(h,w) patch_embed->(ceil(h/patch_size),ceil(w/patch_size)) 
        # z ->(ceil(h/patch_size*8),ceil(w/patch_size*8)) 

        # 2.扩展mask并传入encoder中
        z = self.encoder(x,mask_expanded)  
        # print('z.shape is ',z.shape) # z.shape is  torch.Size([2, 768, 23, 31])
        # print('hw_shape is ',hw_shape) # hw_shape is  (23, 31)
        # print("x.shape is ",x.shape)
        # print("mask_idx is ",mask_idx)
        # import sys
        # sys.exit()
        B, C, H_enc,W_enc = z.shape

        # 3. 通过 1×1 卷积转换为 3 通道（RGB）
        x_rec = self.decoder_conv(z)
        
        # 4.归一化恢复图像（标准化）
        x_rec_norm = (x_rec - x_rec.mean()) / (x_rec.std() + 1e-5)  # 使用标准化

        return x_rec_norm

    def _forward(self, inputs,data_samples):
        """
        Args:
            x=inputs (Tensor): 输入图像，形状 (B, 3, H, W)，例如 H=1400, W=1600
        Returns:
        """
        # 1.计算shape并生成最小的mask(针对z)
        x = inputs 
        # print(x[0][0])
        # import sys
        # sys.exit()
        B, _, h, w = x.shape
        mask = self.generate_mask(B,h,w,patch_size=self.patch_size,mask_ratio=self.mask_ratio,device=x.device)
        mask_expanded = self.expand_mask(mask,h,w,patch_size=self.patch_size)
        # x->(h,w) patch_embed->(ceil(h/patch_size),ceil(w/patch_size)) 
        # z ->(ceil(h/patch_size*8),ceil(w/patch_size*8)) 

        # 2.扩展mask并传入encoder中
        z = self.encoder(x,mask_expanded)  
        # print('z.shape is ',z.shape) # z.shape is  torch.Size([2, 768, 23, 31])
        # print('hw_shape is ',hw_shape) # hw_shape is  (23, 31)
        # print("x.shape is ",x.shape)
        # print("mask_idx is ",mask_idx)
        # import sys
        # sys.exit()
        B, C, H_enc,W_enc = z.shape

        # 3. 通过 1×1 卷积转换为 3 通道（RGB）
        x_rec = self.decoder_conv(z)
        
        # 4.归一化恢复图像（标准化）
        x_rec_norm = (x_rec - x_rec.mean()) / (x_rec.std() + 1e-5)  # 使用标准化
        return x_rec_norm
    
    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        pass