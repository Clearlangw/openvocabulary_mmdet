# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from mmengine.runner.amp import autocast
from torch import Tensor

from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)


def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_

# class CoOpModule(nn.Module):
#     def __init__(self,
#         prompt_length: int=16,
#         prompt_channel: int=768) -> None:
#         #这里是超极简实现，不具备CSC以及固定词init功能
#         super().__init__()

#         self.prompt_length = prompt_length
#         self.prompt_channel = prompt_channel
#         self.coop_prompt = nn.Parameter(torch.zeros(1, self.prompt_length, self.prompt_channel))
#         nn.init.normal_(self.coop_prompt, std=0.02)
    
#     def forward(self, x):
#         return x

class BatchInputsRandomPromptTuning(nn.Module):
    def __init__(self, prompt_size=400, num_prompts=2, in_channels=3):
        super().__init__()
        self.prompt_size = prompt_size
        self.num_prompts = num_prompts
        # prompt参数 shape: [num_prompts, in_channels, prompt_size, prompt_size]
        self.prompts = nn.Parameter(torch.zeros(num_prompts, in_channels, prompt_size, prompt_size))
        nn.init.normal_(self.prompts, std=0.02)

    def forward(self, x):
        # x: [bs, 3, h, w]
        bs, c, h, w = x.shape
        device = x.device
        out = x.clone()
        for i in range(bs):
            used_rects = []
            for j in range(self.num_prompts):
                # 随机采样位置，保证不超界且不重叠
                for _ in range(15):  # 最多尝试15次
                    top = random.randint(0, h - self.prompt_size)
                    left = random.randint(0, w - self.prompt_size)
                    rect = (top, left, top + self.prompt_size, left + self.prompt_size)
                    # 检查是否与已用区域重叠
                    overlap = False
                    for ut, ul, ub, ur in used_rects:
                        if not (rect[2] <= ut or rect[0] >= ub or rect[3] <= ul or rect[1] >= ur):
                            overlap = True
                            break
                    if not overlap:
                        used_rects.append(rect)
                        break
                # 替换输入
                out[i, :, top:top+self.prompt_size, left:left+self.prompt_size] += self.prompts[j]
        return out

class ClusterVisualPromptTuning(nn.Module):
    """视觉特征的 Prompt-Tuning 模块
    
    考虑多尺度特征的空间对应关系，通过聚合bbox区域选择tuning位置，
    并按比例将prompt应用到不同层级的特征上。
    """
    def __init__(self, embed_dim=256, prompt_size=16, cluster_num=4,scale_factors=[1, 4, 16, 64]):
        super().__init__()
        # 视觉 prompt 参数，可以学习
        self.visual_prompts = nn.Parameter(torch.zeros(cluster_num, prompt_size, prompt_size, embed_dim))
        # 初始化
        nn.init.normal_(self.visual_prompts, std=0.02)
        self.prompt_size = prompt_size
        self.cluster_num = cluster_num  # 聚类中心数量
        
        # 定义不同层级的缩放比例
        # 第一层是最细粒度的特征层(高分辨率)，最后一层是最粗粒度的(低分辨率)
        # 例如：如果有4个层级，分辨率比例为[1:4:16:64]，则scale_factors应为[1, 4, 16, 64]
        self.scale_factors = scale_factors  # 从最细粒度层到最粗粒度层
        
    def cluster_reference_points(self, reference_points, max_clusters=4, min_points=3):
        """
        对参考点进行聚类，返回聚类中心
        
        Args:
            reference_points: 形状为 [bs, nq, 4]，值为 (cx, cy, w, h)
            max_clusters: 最大聚类数量
            min_points: 每个聚类最少包含的点数
            
        Returns:
            cluster_centers: 形状为 [bs, n_clusters, 4]，聚类中心
            cluster_weights: 形状为 [bs, n_clusters]，每个聚类的权重
        """
        bs, nq, _ = reference_points.shape
        device = reference_points.device
        
        # 创建返回值容器
        cluster_centers = torch.zeros(bs, self.cluster_num, 4, device=device)
        cluster_weights = torch.zeros(bs, self.cluster_num, device=device)
        
        for b in range(bs):
            # 提取当前batch的参考点
            refs = reference_points[b]  # [nq, 4]
            # 简单的基于距离的聚类
            # 1. 初始化聚类中心为随机选择的max_clusters个点
            n_clusters = min(max_clusters, nq)
            # 随机选择n_clusters个点作为初始聚类中心
            if nq > n_clusters:
                random_indices = torch.randperm(nq, device=refs.device)[:n_clusters]
                centers = refs[random_indices].clone()  # [n_clusters, 4]
            else:
                centers = refs.clone()  # 如果点数不足，使用所有点
            
            # 2. 分配点到最近的聚类
            distances = torch.cdist(refs[:, :2], centers[:, :2])  # [nq, n_clusters]
            cluster_ids = torch.argmin(distances, dim=1)  # [nq]
            
            # 3. 更新聚类中心
            for i in range(n_clusters):
                cluster_points = refs[cluster_ids == i]
                if len(cluster_points) >= min_points:
                    # 更新中心为该聚类所有点的平均值
                    centers[i] = cluster_points.mean(dim=0)
                    # 计算聚类权重（该聚类包含的点数）
                    cluster_weights[b, i] = len(cluster_points)
            
            # 保存结果
            cluster_centers[b, :n_clusters] = centers
        
        # 归一化聚类权重
        cluster_weights = cluster_weights / (cluster_weights.sum(dim=1, keepdim=True).clamp(min=1e-6))
        
        return cluster_centers, cluster_weights
    
    def forward(self, memory, reference_points, spatial_shapes, level_start_index):
        """
        Args:
            memory: 形状为 [bs, nvq, dim]
            reference_points: 形状为 [bs, nq, 4]，值为 (cx, cy, w, h)
            spatial_shapes: 形状为 [num_levels, 2]，表示每层特征图的 (h, w)
            level_start_index: 形状为 [num_levels]，表示每层特征在 memory 中的起始索引
        """
        bs, nvq, dim = memory.shape
        num_levels = spatial_shapes.shape[0]
        
        # 对参考点进行聚类，获取聚类中心和权重
        cluster_centers, cluster_weights = self.cluster_reference_points(
            reference_points, max_clusters=self.cluster_num)
        
        # 创建一个与 memory 相同形状的零张量用于存储 prompt
        prompt_tensor = torch.zeros_like(memory)
        
        # 确定基准层（第一层是最细粒度的特征层）
        base_level = 0
        
        # 对每个聚类中心，计算其在各层特征图中对应的位置并应用 prompt
        for b in range(bs):
            for c_idx, center in enumerate(cluster_centers[b]):
                # 如果该聚类权重为0，跳过
                if cluster_weights[b, c_idx] <= 1e-6:
                    continue
                    
                cx, cy, w, h = center  # 归一化坐标 [0, 1]
                
                # 对每一层特征图
                for lvl in range(num_levels):
                    h_lvl, w_lvl = spatial_shapes[lvl]
                    start_idx = level_start_index[lvl]
                    
                    # 计算该层相对于基准层的缩放比例
                    scale_factor = self.scale_factors[lvl] if lvl < len(self.scale_factors) else 1
                    
                    # 计算聚类中心在当前层的坐标
                    x_coord = int(cx * w_lvl)
                    y_coord = int(cy * h_lvl)
                    
                    # 根据不同层级调整prompt的应用区域大小
                    # 细粒度层(scale_factor小)使用较小的prompt区域，粗粒度层(scale_factor大)使用较大的prompt区域
                    # 这是因为在粗粒度层上，一个像素对应原图的区域更大
                    effective_prompt_size = max(4, int(self.prompt_size * scale_factor / self.scale_factors[-1]))
                    half_size = effective_prompt_size // 2
                    
                    # 计算prompt区域的边界
                    x_min = max(0, x_coord - half_size)
                    x_max = min(w_lvl, x_coord + half_size)
                    y_min = max(0, y_coord - half_size)
                    y_max = min(h_lvl, y_coord + half_size)
                    
                    # 计算在memory中的索引范围并应用prompt
                    for y in range(y_min, y_max):
                        for x in range(x_min, x_max):
                            idx = start_idx + y * w_lvl + x
                            if idx < nvq:  # 确保索引不越界
                                # 计算在prompt中的相对位置
                                # 使用双线性插值来处理不同尺寸的映射
                                p_y_ratio = (y - y_min) / max(1, y_max - y_min - 1)
                                p_x_ratio = (x - x_min) / max(1, x_max - x_min - 1)
                                
                                p_y = min(int(p_y_ratio * (self.prompt_size - 1)), self.prompt_size - 1)
                                p_x = min(int(p_x_ratio * (self.prompt_size - 1)), self.prompt_size - 1)
                                
                                # 应用prompt，并根据聚类权重进行加权
                                prompt_tensor[b, idx] += self.visual_prompts[c_idx, p_y, p_x] * cluster_weights[b, c_idx]
        
        # 将prompt添加到原始memory
        tuned_memory = memory + prompt_tensor
        
        return tuned_memory


class SimpleScoreAdjuster(nn.Module):
    def __init__(self, 
                 bg_feature_dim=1, 
                 os_proxy_dim=1, 
                 init_config=None,
                 # 为nan_to_num设置默认的替换值
                 nan_replace_val=-1e5, 
                 posinf_replace_val=1e5, # 根据您的分数范围调整
                 neginf_replace_val=-1e5 # 根据您的分数范围调整
                 ):
        super().__init__()
        self.bg_feature_dim = bg_feature_dim
        self.os_proxy_dim = os_proxy_dim
        self.nan_replace_val = nan_replace_val
        self.posinf_replace_val = posinf_replace_val
        self.neginf_replace_val = neginf_replace_val

        # --- 门控结构 ---
        self.ig_fc_bs = nn.Linear(self.bg_feature_dim, 1, bias=False)
        self.ig_fc_os_proxy = nn.Linear(self.os_proxy_dim, 1, bias=False)
        self.ig_bias = nn.Parameter(torch.Tensor(1))

        self.ca_fc_bs = nn.Linear(self.bg_feature_dim, 1, bias=False)
        self.ca_fc_os_proxy = nn.Linear(self.os_proxy_dim, 1, bias=False)
        self.ca_bias = nn.Parameter(torch.Tensor(1))
        self.adjustment_scale = nn.Parameter(torch.Tensor([0.5]))

        self.pg_fc_bs = nn.Linear(self.bg_feature_dim, 1, bias=False)
        self.pg_fc_os_proxy = nn.Linear(self.os_proxy_dim, 1, bias=False)
        self.pg_bias = nn.Parameter(torch.Tensor(1))
        
        self._initialize_weights(init_config)

    def _initialize_weights(self, config):
        # ... (初始化代码保持不变) ...
        config = config or {}
        
        nn.init.constant_(self.ig_fc_bs.weight, config.get('w_ig_bs', 0.1))
        nn.init.constant_(self.ig_fc_os_proxy.weight, config.get('w_ig_os_proxy', 0.1))
        nn.init.constant_(self.ig_bias, config.get('b_ig', -1.5))

        nn.init.constant_(self.ca_fc_bs.weight, config.get('w_ca_bs', -0.3))
        nn.init.constant_(self.ca_fc_os_proxy.weight, config.get('w_ca_os_proxy', 0.15))
        nn.init.constant_(self.ca_bias, config.get('b_ca', 0.0))
        with torch.no_grad():
            self.adjustment_scale.fill_(config.get('adj_scale_init', 0.5))

        nn.init.constant_(self.pg_fc_bs.weight, config.get('w_pg_bs', -0.1))
        nn.init.constant_(self.pg_fc_os_proxy.weight, config.get('w_pg_os_proxy', 1.5))
        nn.init.constant_(self.pg_bias, config.get('b_pg', 1.5))


    def forward(self, enc_outputs_class_per_proposal, background_similarity_features):
        # 1. 清理 enc_outputs_class_per_proposal 中的 inf/nan
        # 这是处理问题的关键步骤
        safe_enc_outputs_class = torch.nan_to_num(
            enc_outputs_class_per_proposal, 
            nan=self.nan_replace_val, 
            posinf=self.posinf_replace_val, 
            neginf=self.neginf_replace_val
        )

        # 2. 从清理后的特征中提取代理原始分数 (os_proxy)
        os_proxy, _ = torch.max(safe_enc_outputs_class, dim=-1, keepdim=True)
        # os_proxy 现在应该是有限值

        # 3. 清理 background_similarity_features (以防万一)
        safe_background_similarity = torch.nan_to_num(
            background_similarity_features,
            nan=self.nan_replace_val,
            posinf=self.posinf_replace_val,
            neginf=self.neginf_replace_val
        )
        
        bs_f = safe_background_similarity 
        if bs_f.ndim == 2: 
            if self.bg_feature_dim != 1:
                 raise ValueError(f"background_similarity is 2D, implying bg_feature_dim=1, but self.bg_feature_dim={self.bg_feature_dim}")
            bs_f = bs_f.unsqueeze(-1)
        elif bs_f.ndim == 3 and bs_f.shape[-1] != self.bg_feature_dim:
            raise ValueError(f"background_similarity last dim {bs_f.shape[-1]} != self.bg_feature_dim {self.bg_feature_dim}")
        
        # --- 门控逻辑 ---
        # 所有输入到线性层的都应该是清理后的有限值
        ig_signal = self.ig_fc_bs(bs_f) + self.ig_fc_os_proxy(os_proxy) + self.ig_bias
        ig_gate = torch.sigmoid(ig_signal)

        ca_signal = self.ca_fc_bs(bs_f) + self.ca_fc_os_proxy(os_proxy) + self.ca_bias
        candidate_adjustment_scalar = torch.tanh(ca_signal) * self.adjustment_scale

        pg_signal = self.pg_fc_bs(bs_f) + self.pg_fc_os_proxy(os_proxy) + self.pg_bias
        pg_gate = torch.sigmoid(pg_signal)
        
        # --- 应用调整 ---
        # safe_enc_outputs_class 是有限值，门控值和调整量也是有限值
        adjusted_enc_outputs_class = safe_enc_outputs_class * pg_gate + \
                                     (candidate_adjustment_scalar * ig_gate)
        
        # (可选) 对最终输出再做一次清理，以防万一中间计算引入极值（但理论上激活函数会限制）
        adjusted_enc_outputs_class = torch.nan_to_num(
            adjusted_enc_outputs_class,
            nan=self.nan_replace_val,
            posinf=self.posinf_replace_val,
            neginf=self.neginf_replace_val
        )
        
        return adjusted_enc_outputs_class

class TokenContrastiveLoss(nn.Module):
    """
    针对Token级别的特征进行对比学习的损失函数。

    正样本对定义: 来自同一个batch中不同样本，但在原始token序列中处于相同位置的特征。
    负样本对定义: 所有其他的特征对。

    目标:
    1. 拉近正样本对 (e.g., memory[0, i] 和 memory[1, i]) 的相似度。
    2. 推远负样本对 (e.g., memory[0, i] 和 memory[0, j], memory[0, i] 和 background[1, k]) 的相似度。

    Args:
        temperature (float): 温度系数，用于缩放相似度得分，控制损失函数的锐度。
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        """
        计算文本对比损失。
        """
        # --- 1. 获取维度信息并验证输入 ---
        if text_feat.dim() != 3:
            raise ValueError("Inputs must be 3D tensors: [bs, token_num, feat_dim]")
            
        bs, token_num, feat_dim = text_feat.shape
        # --- 2. 特征展平与标准化 ---
        features = text_feat
        features_flat = features.view(-1, feat_dim)
        features_flat = F.normalize(features_flat, p=2, dim=1)
        device = features.device
        # --- 3. 创建用于识别正负样本的索引 ---
        batch_indices = torch.arange(bs, device=device).view(bs, 1).expand(-1, token_num).reshape(-1)
        #[0000011111]
        token_indices = torch.arange(token_num, device=device).repeat(bs)
        #[012012]
        # --- 4. 计算相似度矩阵并构建正样本掩码 ---
        similarity_matrix = torch.matmul(features_flat, features_flat.T)
        
        # 核心逻辑：当两个token的 'token_indices' 相同，但 'batch_indices' 不同时，它们是正样本对
        same_token_mask = (token_indices.unsqueeze(1) == token_indices.unsqueeze(0))
        diff_batch_mask = (batch_indices.unsqueeze(1) != batch_indices.unsqueeze(0))
        positive_mask = same_token_mask & diff_batch_mask

        # --- 5. 计算InfoNCE损失 ---
        # 从负样本中排除自身 (i=j 的情况)
        self_mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=device)
        
        # 计算 logits
        logits = similarity_matrix / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        # 用极小值填充对角线，使其在softmax中被忽略
        logits.masked_fill_(self_mask, -9e15)

        # 计算 log-softmax 概率
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # 针对每个锚点，计算其所有正样本的平均log-likelihood
        # 使用 clamp(min=1) 防止在 batch_size=1 或没有正样本时出现除以零的错误
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
        
        # 最终损失是所有锚点的平均负log-likelihood
        loss = -mean_log_prob_pos.mean()

        return loss

#留给提取原型的gt
def prepare_gt_info(batch_data_samples, batch_inputs):
    """
    预处理GT信息，转换为适合原型特征提取的格式
    
    Args:
        batch_data_samples: 包含GT信息的样本列表
        batch_inputs: 图像输入张量 [B, C, H, W]
    
    Returns:
        gt_info: 包含相对位置和标签的GT信息字典
    """
    B, C, H, W = batch_inputs.shape
    
    gt_info = {
        'bboxes': [],           # 归一化坐标 [B, num_gt, 4]
        'labels': [],           # 类别标签 [B, num_gt]
        'num_gt_per_image': []  # 每张图像的GT数量
    }
    
    for i, data_sample in enumerate(batch_data_samples):
        gt_instances = data_sample.gt_instances
        
        # 获取GT bbox和labels
        bboxes = gt_instances.bboxes  # [num_gt, 4] - [x1, y1, x2, y2]
        labels = gt_instances.labels  # [num_gt]
        
        # 转换为归一化坐标 [0, 1]
        normalized_bboxes = bboxes.clone()
        normalized_bboxes[:, [0, 2]] /= W  # x坐标归一化
        normalized_bboxes[:, [1, 3]] /= H  # y坐标归一化
        
        # 存储信息
        gt_info['bboxes'].append(normalized_bboxes)
        gt_info['labels'].append(labels)
        # gt_info['pixel_bboxes'].append(bboxes)
        # gt_info['image_shapes'].append([H, W])
        gt_info['num_gt_per_image'].append(len(bboxes))
     
    return gt_info

@MODELS.register_module()
class GroundingDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 use_coop=False,
                 use_fake_coop=False,
                 use_cocoop=False,
                 use_maple=False,
                 coop_init=None,
                 coop_csc=False,
                 background_supp=False, #背景抑制background_suppression，在predecoder那里修改
                 background_supp_mode = None,
                 use_mona = False, #mona微调，使用mona对swintransformer进行微调
                 num_tokens=27,
                 text_contrast = False,
                 use_visual_prompt = False,
                 visual_prompt_size = 16,
                 visual_prompt_clusters = 4,
                 visual_prompt_scale_factors = [1, 4, 16, 64],
                 visual_prompt_max_ref_points = 30,
                 use_random_input_prompt = False,
                 #shine的参数写在了language_model那里去了
                 use_text_attn_enhance=False,
                 text_attn_enhance_refpath=None,
                 use_seeker_adapter=False,
                 use_text_consistency_loss=False,
                 use_visual_seeker=False, #这个控制是否使用gt
                 *args,
                 use_autocast=False,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        assert sum([use_coop, use_cocoop, use_maple]) <= 1, "Only one of 'use_coop', 'use_cocoop', or 'use_maple' can be True at a time."
        #text端：
        self.use_coop=use_coop
        self.use_fake_coop = use_fake_coop
        #TODO:这两个确实还没实现
        self.use_cocoop=use_cocoop
        self.use_maple=use_maple

        self.coop_init=coop_init
        self.coop_csc =coop_csc #是否每一类均有不同的上下文参数，仅参考coop原论文
        #coop相关参数
        #TODO:实际to_enhance_text_prompts有前后缀的实现，可以merge
        self.coop_prompt_length=self.coop_n_ctx = 16 #可训练参数长度(左命名参考mrgdino，右名参考coop原论文)
        self.coop_prompt_channel=self.coop_ctx_dim  = 768
        self.num_tokens = num_tokens #用于告诉coop微调词的数量
        #其余参数
        self.text_contrast = text_contrast
        self.use_text_attn_enhance = use_text_attn_enhance
        self.text_attn_enhance_refpath = text_attn_enhance_refpath
        #vis端：swin微调参数 实际上并没有在这里起作用
        self.use_mona = use_mona
        # 这里的prompt不是vpt
        self.use_visual_prompt = use_visual_prompt
        self.visual_prompt_size = visual_prompt_size
        self.visual_prompt_clusters = visual_prompt_clusters
        self.visual_prompt_scale_factors = visual_prompt_scale_factors
        self.visual_prompt_max_ref_points = visual_prompt_max_ref_points
        self.use_random_input_prompt = use_random_input_prompt
        #VL交互：
        self.use_seeker_adapter = use_seeker_adapter
        self.use_text_consistency_loss = use_text_consistency_loss
        self.use_visual_seeker = use_visual_seeker
        self.background_supp = background_supp
        self.background_supp_mode = background_supp_mode
        self.classnames = None #
        #原型测试参数 #TODO
        
        self.accumulated_background_similarity = []
        self.accumulated_original_scores = []

        self._special_tokens = '. '
        self.use_autocast = use_autocast
        super().__init__(*args, **kwargs)
        

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        #初始化coop
        if self.use_coop:
            self.language_model.add_coop_prompt(self.coop_prompt_length,self.coop_prompt_channel,self.coop_init,self.coop_csc)
        if self.use_fake_coop:
            self.fake_coop = nn.Parameter(torch.zeros(1, self.num_tokens, self.coop_prompt_channel))
            nn.init.normal_(self.fake_coop, std=0.02)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)
        if self.background_supp and self.background_supp_mode == 'lstm_gating':
            self.score_adjuster = SimpleScoreAdjuster(
                bg_feature_dim=1, # 假设 background_similarity 输入是聚合后的标量
                os_proxy_dim=1,   # 因为 os_proxy 是通过 max(...) keepdim=True 得到的
                init_config={ # 您可以按需调整这些初始化值
                    'w_ig_bs': 0.1, 'w_ig_os_proxy': 0.1, 'b_ig': -1.5,
                    'w_ca_bs': -0.3, 'w_ca_os_proxy': 0.15, 'b_ca': 0.0, 'adj_scale_init': 0.5,
                    'w_pg_bs': -0.1, 'w_pg_os_proxy': 1.5, 'b_pg': 1.5,
                }
            )
        if self.use_text_attn_enhance:
            self.text_attn_enhancer = MultiheadAttention(embed_dims=256,num_heads=8,batch_first=True)

        if self.text_contrast:
            self.text_contrast_loss = TokenContrastiveLoss()
        # 添加视觉prompt-tuning模块
        if self.use_visual_prompt:
            self.visual_prompt_tuning = ClusterVisualPromptTuning(
                embed_dim=self.embed_dims,
                prompt_size=self.visual_prompt_size,
                cluster_num=self.visual_prompt_clusters,
                scale_factors=self.visual_prompt_scale_factors
            )
        if self.use_random_input_prompt:
            self.random_input_prompt_tuning = BatchInputsRandomPromptTuning()

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            #print('caption_string is ',caption_string)   #pedestrian. people. bicycle. car. van. truck. tricycle. awning-tricycle. bus. motor.
            # import sys
            # sys.exit()
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt') #输出27个token
            # print("___________________________________________________")
            # word_ids = tokenized.word_ids()
            # # 打印结果
            # tokens = self.language_model.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])
            # for i, (token, word_id) in enumerate(zip(tokens, word_ids)):
            #     print(f"Token {i}: {token}, belongs to word index: {word_id}")

            # Token 0: [CLS], belongs to word index: None
            # Token 1: pedestrian, belongs to word index: 0
            # Token 2: ., belongs to word index: 1
            # Token 3: people, belongs to word index: 2
            # Token 4: ., belongs to word index: 3
            # Token 5: bicycle, belongs to word index: 4
            # Token 6: ., belongs to word index: 5
            # Token 7: car, belongs to word index: 6
            # Token 8: ., belongs to word index: 7
            # Token 9: van, belongs to word index: 8
            # Token 10: ., belongs to word index: 9
            # Token 11: truck, belongs to word index: 10
            # Token 12: ., belongs to word index: 11
            # Token 13: tri, belongs to word index: 12
            # Token 14: ##cycle, belongs to word index: 12
            # Token 15: ., belongs to word index: 13
            # Token 16: aw, belongs to word index: 14
            # Token 17: ##ning, belongs to word index: 14
            # Token 18: -, belongs to word index: 15
            # Token 19: tri, belongs to word index: 16
            # Token 20: ##cycle, belongs to word index: 16
            # Token 21: ., belongs to word index: 17
            # Token 22: bus, belongs to word index: 18
            # Token 23: ., belongs to word index: 19
            # Token 24: motor, belongs to word index: 20
            # Token 25: ., belongs to word index: 21
            # Token 26: [SEP], belongs to word index: None
            # import sys
            # sys.exit()

            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption) #名词提取
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    ##为了增加文本特征提前和视觉交互，就重写了这个函数
    #TODO:第二处注入，需要考虑将gt也注入到swin中间的函数兼容
    # def extract_feat(self, batch_inputs: Tensor, text_inputs: Tensor = None) -> Tuple[Tensor]:
    #     """Extract features.

    #     Args:
    #         batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).
    #         text_inputs (Tensor, optional): Text tensor, has shape (bs, L, D). Defaults to None.
    #     Returns:
    #         tuple[Tensor]: Tuple of feature maps from neck. Each feature map
    #         has shape (bs, dim, H, W).
    #     """
    #     if text_inputs is None:
    #         x = self.backbone(batch_inputs)
    #     else:
    #         x = self.backbone(batch_inputs, text_inputs)
    #     if self.use_text_consistency_loss:
    #         x, consistency_loss = x
    #     if self.with_neck:
    #         x = self.neck(x)
    #     if self.use_text_consistency_loss:
    #         return x, consistency_loss
    #     else:
    #         return x
    def extract_feat(self, batch_inputs: Tensor, text_inputs: Tensor = None, gt_info: dict = None) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).
            text_inputs (Tensor, optional): Text tensor, has shape (bs, L, D). Defaults to None.
            gt_info (dict, optional): GT information for prototype enhancement. Defaults to None.
        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        # 根据输入情况调用backbone
        if text_inputs is None and gt_info is None:
            # 情况1: 无额外输入，保持原有行为
            x = self.backbone(batch_inputs)
        elif text_inputs is not None and gt_info is None:
            # 情况2: 只输入text_inputs，保持原有行为
            x = self.backbone(batch_inputs, text_inputs)
        elif text_inputs is None and gt_info is not None:
            # 情况3: 只输入gt_info
            x = self.backbone(batch_inputs, gt_info=gt_info)
        else:
            # 情况4: 两者均输入
            x = self.backbone(batch_inputs, text_inputs, gt_info=gt_info)
        
        if self.use_text_consistency_loss:
            x, consistency_loss = x
        if self.with_neck:
            x = self.neck(x)
        if self.use_text_consistency_loss:
            return x, consistency_loss
        else:
            return x

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        #这里是总的
        # The forward procedure of the transformer is defined as:
        # 'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        # for name,parameter in self.named_parameters():
        #     if "lora" in name:
        #         print(f"layer :  {name}  is  {parameter.requires_grad}")
        # import sys
        # sys.exit()
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        #  dict: The dictionary of encoder outputs, which includes the
        #  `memory` of the encoder output.
        background_text_feat = None
        background_text_token_mask = None
        # --- 背景抑制逻辑 ---
        if self.background_supp:
            background_text_dict = torch.load('/home/wuke_2024/ov202503/mmdetection/ov_text_feature/back_dict.pth',map_location=memory.device)
            bs,_,_ = memory.shape
            for key in background_text_dict.keys():
                #针对不同的key
                if background_text_dict[key].dim() == 2:
                    background_text_dict[key] = background_text_dict[key][0]
                    background_text_dict[key] = background_text_dict[key].unsqueeze(0).expand(bs, -1)
                elif background_text_dict[key].dim() == 3:
                    background_text_dict[key] = background_text_dict[key][0]
                    if key == 'dot_mask':
                        background_text_dict[key][0]=False
                        background_text_dict[key][-1]=False
                    background_text_dict[key] = background_text_dict[key].unsqueeze(0).expand(bs, -1, -1)
                elif background_text_dict[key].dim() == 1:
                    # dot_mask 需要保证首位和末尾为False
                    background_text_dict[key][0]=False
                    background_text_dict[key][-1]=False
                    background_text_dict[key] = background_text_dict[key].unsqueeze(0).expand(bs, -1)

            # 扩展batch维度
            background_text_feat = background_text_dict['embedded']
            background_text_feat=self.text_feat_map(background_text_feat)
            background_text_token_mask = background_text_dict['dot_mask']
            with torch.no_grad():
                _, background_text_feat = self.encoder(
                    query=feat,
                    query_pos=feat_pos,
                    key_padding_mask=feat_mask,  # for self_attn
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    # for text encoder
                    memory_text=background_text_feat,
                    text_attention_mask=~background_text_token_mask,
                    position_ids=background_text_dict['position_ids'],
                    text_self_attention_masks=background_text_dict['masks'])
        #     print(background_text_dict['masks'].shape) #输出数据
        #     print(background_text_dict['text_token_mask'].shape) #输出数据
        #     print(type(background_text_dict['text_token_mask'])) #输出类型
        # import sys
        # sys.exit()
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            background_text_feat=background_text_feat,
            background_text_token_mask=background_text_token_mask,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        background_text_feat: Optional[Tensor] = None,
        background_text_token_mask: Optional[Tensor] = None,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)

        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        num_all_proposals = output_proposals.shape[1]
        original_scores = enc_outputs_class.max(-1)[0]
        alpha = 0.5             # 用于 'score_modulation',最初为0.5
        m_factor = 2               # 用于 'reranking',最初为2
        beta = 0.5               # 用于 'reranking',最初为0.5  
        similarity_threshold = -4 # 用于 'gating' (需要根据点积的范围调整) 最开始为10.0
        penalty_value = 1e5       # 用于 'gating'
        if self.background_supp and background_text_feat is not None:

            ori_background_similarity = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, background_text_feat,
                                    background_text_token_mask) #这里偷了懒
            background_similarity = ori_background_similarity.max(-1)[0]
            # # 验证相似度(zero shot下示例)
            # print(f'background_similarity is',background_similarity) 
            # print(f'original_scores is',original_scores) 
            # background_mask = background_similarity > similarity_threshold
            # print(f'background_mask is',background_mask)
            # print(f'background_similarity min: {background_similarity.min().item():.4f}, max: {background_similarity.max().item():.4f}')
            # print(f'original_scores min: {original_scores.min().item():.4f}, max: {original_scores.max().item():.4f}')
            # print(f'background_mask True count: {background_mask.sum().item()}')
            # print(f'background_mask False count: {(~background_mask).sum().item()}')
            # self.accumulated_background_similarity.append(background_similarity.cpu().numpy())
            # self.accumulated_original_scores.append(original_scores.cpu().numpy())

            if self.background_supp_mode == 'score_modulation':
                # 方案1: 用背景相似度调整原始得分
                modulated_scores = original_scores - alpha * (background_similarity - similarity_threshold)
                topk_indices = torch.topk(modulated_scores, k=self.num_queries, dim=1)[1]
                # print(f"使用score_modulation。原始最高分: {original_scores.max().item()}, 调整后最高分: {modulated_scores.max().item()}")

            elif self.background_supp_mode == 'reranking':
                # 方案2: 选择top-M, 然后使用背景相似度进行重排序
                num_queries_k = self.num_queries
                num_queries_m = min(num_queries_k * m_factor, num_all_proposals)

                top_m_original_scores, top_m_indices = torch.topk(
                    original_scores, k=num_queries_m, dim=1)
                
                top_m_output_memory_feat = torch.gather(
                    output_memory, 1,
                    top_m_indices.unsqueeze(-1).repeat(1, 1, c))

                # 计算这M个候选框与背景token的最大相似度
                # top_m_output_memory_feat: (bs, M, c_feat)
                # proposal_m_bg_token_sim_matrix 形状: (bs, M, num_bg_tokens)

                proposal_m_bg_token_sim_matrix =  self.bbox_head.cls_branches[
                self.decoder.num_layers](top_m_output_memory_feat, background_text_feat,
                                        background_text_token_mask)
                background_similarity_m = torch.max(proposal_m_bg_token_sim_matrix, dim=-1)[0] # (bs, M)
                
                reranked_scores_m = top_m_original_scores - beta * background_similarity_m
                _, final_topk_indices_in_m = torch.topk(
                    reranked_scores_m, k=num_queries_k, dim=1)
                topk_indices = torch.gather(top_m_indices, 1, final_topk_indices_in_m)
                # print(f"使用reranking。M={num_queries_m}, K={num_queries_k}")

            elif self.background_supp_mode == 'gating':
                # 方案3: 惩罚背景相似度高于阈值的候选框
                background_mask = background_similarity > similarity_threshold
                penalized_scores = original_scores - background_mask.float() * penalty_value
                topk_indices = torch.topk(penalized_scores, k=self.num_queries, dim=1)[1]
                # print(f"使用gating。被掩码的大致数量: {background_mask.sum().item() / bs}")
            elif self.background_supp_mode == 'gating_v2':
                # 方案4：避免前景分数本身非常高的被抑制（），使用非学习参数的固定参数控制门控行为
                # --- 在 pre_decoder 方法内部，计算完 original_scores 和 background_similarity 之后 ---
                # 乘法门控机制的参数 (这些最好是类的属性或者可配置的)
                self.bg_sensitivity_pivot = -4.0  # 背景相似度抑制起始点 (接近 full_train 背景相似度的75%分位数)
                self.bg_sensitivity_steepness = 2.0 # 当背景相似度增加时，门关闭的陡峭程度
                self.score_rescue_pivot = -1.0     # 原始得分"救援"起始点 (full_train 长尾分布中表现开始显著变好的区域)
                self.score_rescue_steepness = 1.5  # 高原始得分能够阻止或减缓门关闭的强度
                self.min_gate_multiplier = 0.15    # 门控乘数的最小值，防止得分完全变为零
                # 1. 计算背景相似度带来的抑制趋势
                #    如果 background_similarity 比 bg_sensitivity_pivot 更正 (更像背景)，suppression_raw 为正且增大
                suppression_raw = (background_similarity - self.bg_sensitivity_pivot) * self.bg_sensitivity_steepness
                # 2. 计算原始得分带来的救援趋势
                #    如果 original_scores 比 score_rescue_pivot 更正 (得分更高)，rescue_effect 为正且增大
                rescue_effect = (original_scores - self.score_rescue_pivot) * self.score_rescue_steepness
                # 3. 结合抑制和救援趋势，得到门控控制信号
                #    我们希望：如果抑制趋势强且救援趋势弱，则信号为正 (倾向于关闭门，即抑制)
                #    如果抑制趋势弱或救援趋势强，则信号为负 (倾向于打开门，即不抑制或少抑制)
                gate_control_signal = suppression_raw - rescue_effect
                # 4. 使用 tanh 函数将控制信号映射到 (-1, 1) 区间
                #    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
                #    如果 gate_control_signal 非常正 (强抑制信号)，tanh -> 1
                #    如果 gate_control_signal 非常负 (弱抑制或强救援)，tanh -> -1
                suppression_factor_tanh = torch.tanh(gate_control_signal)
                # 5. 将 tanh 输出的抑制因子 (-1到1) 转换为最终的门控乘数 (min_gate_multiplier 到 1)
                #    如果 suppression_factor_tanh = -1 (弱抑制或者强救援)，则 gate_multiplier = 2 - min_gate_multiplier,实现增强
                #    如果 suppression_factor_tanh = 1 (最大抑制)，则 gate_multiplier = min_gate_multiplier
                gate_multiplier = 1.0 - suppression_factor_tanh * (1.0 - self.min_gate_multiplier)
                # 6. 应用门控，得到调整后的分数
                final_scores_for_selection = original_scores * gate_multiplier
                # 7. 使用调整后的分数进行 top-k 选择
                topk_indices = torch.topk(final_scores_for_selection, k=self.num_queries, dim=1)[1]
            elif self.background_supp_mode == 'lstm_gating':
                # 方案5：使用可学习的参数控制background分数对前景分数的抑制程度
                adjusted_enc_outputs_class = self.score_adjuster(
                    enc_outputs_class,
                    background_similarity
                )
                topk_indices = torch.topk(adjusted_enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
            else: 
                topk_indices = torch.topk(original_scores, k=self.num_queries, dim=1)[1]
        else:
            # 没有背景抑制或未提供 background_text_feat
            topk_indices = torch.topk(original_scores, k=self.num_queries, dim=1)[1]
            # if self.background_supp and background_text_feat is None:
            #    print("背景抑制已启用, 但 background_text_feat 为 None。跳过。")
        # --- 背景抑制逻辑结束 ---

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.

        # topk_indices = torch.topk(
        #     enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        if self.background_supp and self.background_supp_mode == 'lstm_gating':
            topk_score = torch.gather(
                adjusted_enc_outputs_class, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        else:
            topk_score = torch.gather(
                enc_outputs_class, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()
        if self.text_contrast:
            if not self.background_supp:
                text_contrast_loss = self.text_contrast_loss(memory_text)
            else:
                ft = torch.cat([background_text_feat, memory_text], dim=1)
                text_contrast_loss = self.text_contrast_loss(ft)
        
        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()   
        # print("reference_points.shape is",reference_points.shape)
        # 应用视觉prompt-tuning
        if self.use_visual_prompt:
            level_start_index = torch.cat((
                spatial_shapes.new_zeros((1, )),  # (num_level)
                spatial_shapes.prod(1).cumsum(0)[:-1])) #每段开头索引
            visual_prompt_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_indices[:,:self.visual_prompt_max_ref_points].unsqueeze(-1).repeat(1, 1, 4))
            # print("visual_prompt_coords_unact.shape is",visual_prompt_coords_unact.shape)
            # print("visual_prompt_coords_unact is",visual_prompt_coords_unact)
            # import sys
            # sys.exit()
            visual_prompt_coords = visual_prompt_coords_unact.sigmoid()
            memory = self.visual_prompt_tuning(
                memory, visual_prompt_coords, spatial_shapes, level_start_index)
        
        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        if self.text_contrast and self.training:
            head_inputs_dict['text_contrast_loss'] = text_contrast_loss
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        if self.use_random_input_prompt:
            batch_inputs = self.random_input_prompt_tuning(batch_inputs)
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        #TODO:第一处注入，可以考虑将gt_labels，包含bboxes注入到swin中间，方便形成原型，记得确定一下gt_instances的信息格式
        # >>> gt_instances = InstanceData(metainfo=img_meta)
        # >>> gt_instances.bboxes = torch.rand((5, 4))
        # >>> gt_instances.labels = torch.rand((5,))

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]
        if self.use_visual_seeker:
            gt_info = prepare_gt_info(batch_data_samples, batch_inputs)
        else:
            gt_info = None
        # import pdb; pdb.set_trace()
        # print('tokens_positive' in batch_data_samples[0]) #False
        # print(text_prompts[0]) #('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
        # import sys
        # sys.exit()
        assert 'tokens_positive' not in batch_data_samples[0], \
    "Error: tokens_positive should not exist in batch_data_samples[0] during ov-detection"
        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                #TODO:实现有coop填充的token化，之后需要删除没必要的影响
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                #
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            # print('text_prompts are',text_prompts) #(bs,class)
            # import sys
            # sys.exit()
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                # print('tokenized is ',tokenized) #27个token，why tokenizer的正常行为，会有映射指向token和单词的关系
                # print('tokenized[input_ids] is ',tokenized['input_ids'].shape) #27
                #print('caption_string is ',caption_string) #pedestrian. people. bicycle. car. van. truck. tricycle. awning-tricycle. bus. motor.
                #print("tokens_positive is",tokens_positive) #tokens_positive is [[[0, 10]], [[12, 18]], [[20, 27]], [[29, 32]], [[34, 37]], [[39, 44]], [[46, 54]], [[56, 71]], [[73, 76]], [[78, 83]]]
                #tokens_positive:char哪些位置有效
                # import sys
                # sys.exit()
                new_text_prompts = [caption_string] * len(batch_inputs)
                # print("gt labels are ",gt_labels) #(bs,每个label)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    #positive_map_label_to_token, positive_map 
                    positive_maps.append(positive_map)
                # print('positive maps: ',positive_maps) #positive_map 
                # # print('positive maps[0]: ',positive_maps[0])
                # print('positive maps[0].shape: ',positive_maps[0].shape) #[gt,256]
                # print('positive maps[0][0]: ',positive_maps[0][0])
                # print('new_tokens_positive: ',new_tokens_positive)
                # import sys
                # sys.exit()
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)
            # print('new_text_prompts are: ',new_text_prompts)
            # new_text_prompts are:  ['pedestrian. people. bicycle. car. van. truck. tricycle. awning-tricycle. bus. motor. ', 
            # 'pedestrian. people. bicycle. car. van. truck. tricycle. awning-tricycle. bus. motor. ']
            # import sys
            # sys.exit()

        # if self.use_coop:
        #     #把token替换为这部分
        text_dict = self.language_model(new_text_prompts)

        # input_ids 负责表示输入文本的内容， position_ids 负责提供词元的位置信息，，每个token在每次词中间的位置
        # 而 mask 则负责处理填充问题，确保模型能够正确处理批量输入每个词只关心自己部分的token
        # for key in text_dict.keys():
        #     print(f"text_dict's {key} is {text_dict[key]}")
        #     print("*"*100)
        #     print(f"shape of text_dict's {key} is {text_dict[key].shape}")
        # import sys
        # sys.exit()

        # #TODO：Coop在GDINO第四处实现，文本还原为原本的编码文本特征
        # if self.coop:
        #      text_dict['embedded'] = torch.cat((text_dict['embedded'][:,:1], text_dict['embedded'][:,1+self.coop_prompt_length:]), dim=1)
        if self.use_fake_coop:
            fake_coop = self.fake_coop.repeat(text_dict['embedded'].size()[0], 1, 1)
            text_dict['embedded'] = fake_coop+text_dict['embedded']
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        if self.use_text_attn_enhance:
            text_bs,_,_ = text_dict['embedded'].shape
            concept_ft = torch.load(self.text_attn_enhance_refpath,map_location=text_dict['embedded'].device)
            # print(concept_ft)
            # print(concept_ft.shape)
            # import sys
            # sys.exit()
            if concept_ft.dim() == 2:
                concept_ft = concept_ft.unsqueeze(0).expand(text_bs, -1, -1)
            elif concept_ft.dim() == 3:
                concept_ft = concept_ft[0].unsqueeze(0).expand(text_bs, -1, -1)
            else:
                print(f"concept_ft.shape is {concept_ft}")
                print("error in the text_attn_enhancer shape")
                import sys
                sys.exit()
            concept_ft = self.text_feat_map(concept_ft)
            text_dict['embedded'] = self.text_attn_enhancer(query=text_dict['embedded'],key=concept_ft,value=concept_ft)



        # #TODO：Coop早GDINO第五处实现，还原att
        # if self.coop:
        #     position_ids = torch.cat((position_ids[:,:1], position_ids[:,17:]), dim=1)
            
        #     text_self_attention_masks = torch.cat((text_self_attention_masks[:,:1], text_self_attention_masks[:,17:]),dim=1)
            
        #     text_self_attention_masks = torch.cat((text_self_attention_masks[:,:,:1], text_self_attention_masks[:,:,17:]),dim=2)
        # print(self.text_feat_map)#linear[768,256]
        # print('text_dict[embedded] is ',text_dict['embedded'])#没有归一化的一个变量
        # print('text_dict[embedded] shape is ',text_dict['embedded'].shape) #[bs=2,token=27,dim=256]
        # import sys
        # sys.exit()
        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        # print(f"self.use_seeker_adapter is {self.use_seeker_adapter}")
        # print(f"text_dict['embedded'] is {text_dict['embedded']}")
        # import sys
        # sys.exit()
        if self.use_autocast:
            with autocast(enabled=True):
                if self.use_seeker_adapter:
                    if self.use_text_consistency_loss:
                        visual_features, consistency_loss = self.extract_feat(batch_inputs,text_dict['embedded'],gt_info=gt_info) 
                    else:
                        visual_features = self.extract_feat(batch_inputs,text_dict['embedded'],gt_info=gt_info) 
                else:
                    visual_features = self.extract_feat(batch_inputs,gt_info=gt_info)
        else:
            if self.use_seeker_adapter:
                if self.use_text_consistency_loss:
                    visual_features, consistency_loss = self.extract_feat(batch_inputs,text_dict['embedded'],gt_info=gt_info)
                else:
                    visual_features = self.extract_feat(batch_inputs,text_dict['embedded'],gt_info=gt_info)
            else:
                visual_features = self.extract_feat(batch_inputs,gt_info=gt_info)
        # print("visual_features is",visual_features)
        # print(len(visual_features))
        # # print("visual_features.shape is",visual_features.shape)
        # print("visual_features[0] is",visual_features[0])
        # print("visual_features[0].shape is",visual_features[0].shape)
        # print("visual_features[0][0] is",visual_features[0][0])
        # print("visual_features[0][0].shape is",visual_features[0][0].shape)
        # import sys
        # sys.exit()
        #emb-t [2,27,256]
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)
        text_contrast_loss = head_inputs_dict.pop('text_contrast_loss', None)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        if self.text_contrast and self.training:
            losses['text_contrast_loss'] = text_contrast_loss
        if self.use_text_consistency_loss:
            losses['text_consistency_loss'] = consistency_loss
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        if self.use_random_input_prompt:
            batch_inputs = self.random_input_prompt_tuning(batch_inputs)
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)
        #Warning:原来预测的时候visual_feats是在这里提取的
        # image feature extraction
        # visual_feats = self.extract_feat(batch_inputs)
        # print(isinstance(text_prompts[0], list)) #False
        # print(text_prompts[0]) #经典
        # import sys
        # sys.exit()
        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                # print(f"text_prompts_once is {text_prompts_once}")
                # import sys
                # sys.exit()
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.use_fake_coop:
                    fake_coop = self.fake_coop.repeat(text_dict['embedded'].size()[0], 1, 1)
                    text_dict['embedded'] = fake_coop+text_dict['embedded']
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])
                if self.use_text_attn_enhance:
                    text_bs,_,_ = text_dict['embedded'].shape
                    concept_ft = torch.load(self.text_attn_enhance_refpath,map_location=text_dict['embedded'].device)
                    # print(concept_ft)
                    # print(concept_ft.shape)
                    # import sys
                    # sys.exit()
                    if concept_ft.dim() == 2:
                        concept_ft = concept_ft.unsqueeze(0).expand(text_bs, -1, -1)
                    elif concept_ft.dim() == 3:
                        concept_ft = concept_ft[0].unsqueeze(0).expand(text_bs, -1, -1)
                    else:
                        print("error in the text_attn_enhancer shape")
                        import sys
                        sys.exit()
                    concept_ft = self.text_feat_map(concept_ft)
                    text_dict['embedded'] = self.text_attn_enhancer(query=text_dict['embedded'],key=concept_ft,value=concept_ft)
                if self.use_seeker_adapter:
                    if self.use_text_consistency_loss:
                        visual_feats, consistency_loss = self.extract_feat(batch_inputs,text_dict['embedded'])
                    else:
                        visual_feats = self.extract_feat(batch_inputs,text_dict['embedded'])
                else:
                    visual_feats = self.extract_feat(batch_inputs)
                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            #还是走这里的
            # print(f"text_prompts_once is {text_prompts_once}")
            # import sys
            # sys.exit()
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.use_fake_coop:
                # print(f'self.use_fake_coop is {self.use_fake_coop}')
                # import sys
                # sys.exit()
                fake_coop = self.fake_coop.repeat(text_dict['embedded'].size()[0], 1, 1)
                text_dict['embedded'] = fake_coop+text_dict['embedded']
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])
            if self.use_text_attn_enhance:
                text_bs,_,_ = text_dict['embedded'].shape
                concept_ft = torch.load(self.text_attn_enhance_refpath,map_location=text_dict['embedded'].device)
                # print(concept_ft)
                # print(concept_ft.shape)
                # import sys
                # sys.exit()
                if concept_ft.dim() == 2:
                    concept_ft = concept_ft.unsqueeze(0).expand(text_bs, -1, -1)
                elif concept_ft.dim() == 3:
                    concept_ft = concept_ft[0].unsqueeze(0).expand(text_bs, -1, -1)
                else:
                    print("error in the text_attn_enhancer shape")
                    import sys
                    sys.exit()
                concept_ft = self.text_feat_map(concept_ft)
                text_dict['embedded'] = self.text_attn_enhancer(query=text_dict['embedded'],key=concept_ft,value=concept_ft)
            if self.use_seeker_adapter:
                if self.use_text_consistency_loss:
                    visual_feats, consistency_loss = self.extract_feat(batch_inputs,text_dict['embedded'])
                else:
                    visual_feats = self.extract_feat(batch_inputs,text_dict['embedded'])
            else:
                visual_feats = self.extract_feat(batch_inputs)
            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]
            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
    
    # 仅用于测试FPS，GFLOPS
    def forward_dummy(self, inputs, data_samples_dict_list):
        batch_data_samples = dict_list_to_data_samples_list(data_samples_dict_list)
        rescale = True
        batch_inputs = inputs
        if self.use_random_input_prompt:
            batch_inputs = self.random_input_prompt_tuning(batch_inputs)
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        # visual_feats = self.extract_feat(batch_inputs)
        # print(isinstance(text_prompts[0], list)) #False
        # print(text_prompts[0]) #经典
        # import sys
        # sys.exit()
        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                # print(f"text_prompts_once is {text_prompts_once}")
                # import sys
                # sys.exit()
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.use_fake_coop:
                    fake_coop = self.fake_coop.repeat(text_dict['embedded'].size()[0], 1, 1)
                    text_dict['embedded'] = fake_coop+text_dict['embedded']
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])
                if self.use_text_attn_enhance:
                    text_bs,_,_ = text_dict['embedded'].shape
                    concept_ft = torch.load(self.text_attn_enhance_refpath,map_location=text_dict['embedded'].device)
                    # print(concept_ft)
                    # print(concept_ft.shape)
                    # import sys
                    # sys.exit()
                    if concept_ft.dim() == 2:
                        concept_ft = concept_ft.unsqueeze(0).expand(text_bs, -1, -1)
                    elif concept_ft.dim() == 3:
                        concept_ft = concept_ft[0].unsqueeze(0).expand(text_bs, -1, -1)
                    else:
                        print("error in the text_attn_enhancer shape")
                        import sys
                        sys.exit()
                    concept_ft = self.text_feat_map(concept_ft)
                    text_dict['embedded'] = self.text_attn_enhancer(query=text_dict['embedded'],key=concept_ft,value=concept_ft)
                if self.use_seeker_adapter:
                    if self.use_text_consistency_loss:
                        visual_feats, consistency_loss = self.extract_feat(batch_inputs,text_dict['embedded'])
                    else:
                        visual_feats = self.extract_feat(batch_inputs,text_dict['embedded'])
                else:
                    visual_feats = self.extract_feat(batch_inputs)
                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            #还是走这里的
            # print(f"text_prompts_once is {text_prompts_once}")
            # import sys
            # sys.exit()
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.use_fake_coop:
                # print(f'self.use_fake_coop is {self.use_fake_coop}')
                # import sys
                # sys.exit()
                fake_coop = self.fake_coop.repeat(text_dict['embedded'].size()[0], 1, 1)
                text_dict['embedded'] = fake_coop+text_dict['embedded']
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])
            if self.use_text_attn_enhance:
                text_bs,_,_ = text_dict['embedded'].shape
                concept_ft = torch.load(self.text_attn_enhance_refpath,map_location=text_dict['embedded'].device)
                # print(concept_ft)
                # print(concept_ft.shape)
                # import sys
                # sys.exit()
                if concept_ft.dim() == 2:
                    concept_ft = concept_ft.unsqueeze(0).expand(text_bs, -1, -1)
                elif concept_ft.dim() == 3:
                    concept_ft = concept_ft[0].unsqueeze(0).expand(text_bs, -1, -1)
                else:
                    print("error in the text_attn_enhancer shape")
                    import sys
                    sys.exit()
                concept_ft = self.text_feat_map(concept_ft)
                text_dict['embedded'] = self.text_attn_enhancer(query=text_dict['embedded'],key=concept_ft,value=concept_ft)
            if self.use_seeker_adapter:
                if self.use_text_consistency_loss:
                    visual_feats, consistency_loss = self.extract_feat(batch_inputs,text_dict['embedded'])
                else:
                    visual_feats = self.extract_feat(batch_inputs,text_dict['embedded'])
            else:
                visual_feats = self.extract_feat(batch_inputs)
            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]
            # print("*"*100)
            # print("pass the text_dict")
            # print("*"*100)
            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)

            # print("?"*100)
            # print("pass the forward_transformer")
            # print("?"*100)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)
            # print("!"*100)
            # print("pass the bbox_head")
            # print("!"*100)
        # for data_sample, pred_instances, entity, is_rec_task in zip(
        #         batch_data_samples, results_list, entities, is_rec_tasks):
        #     if len(pred_instances) > 0:
        #         label_names = []
        #         for labels in pred_instances.labels:
        #             if is_rec_task:
        #                 label_names.append(entity)
        #                 continue
        #             if labels >= len(entity):
        #                 warnings.warn(
        #                     'The unexpected output indicates an issue with '
        #                     'named entity recognition. You can try '
        #                     'setting custom_entities=True and running '
        #                     'again to see if it helps.')
        #                 label_names.append('unobject')
        #             else:
        #                 label_names.append(entity[labels])
        #         # for visualization
        #         pred_instances.label_names = label_names
        #     data_sample.pred_instances = pred_instances
        # return data_samples_list_to_dict_list(batch_data_samples)
        return torch.tensor(0)

from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData,PixelData
import torch
import numpy as np

def data_samples_list_to_dict_list(data_samples):
    """DetDataSample对象列表转dict列表"""
    return [datasample_to_dict(ds) for ds in data_samples]

def dict_list_to_data_samples_list(dict_list):
    """dict列表转DetDataSample对象列表"""
    return [dict_to_datasample(d) for d in dict_list]


def datasample_to_dict(obj):
    """递归将 DetDataSample/InstanceData/PixelData 转为 dict。"""
    # DetDataSample
    if isinstance(obj, DetDataSample):
        result = {}
        # meta信息
        if hasattr(obj, 'metainfo'):
            result['metainfo'] = dict(obj.metainfo)
        # data fields
        for k in obj._data_fields:
            v = getattr(obj, k)
            result[k] = datasample_to_dict(v)
        return result
    # InstanceData 或 PixelData
    elif isinstance(obj, (InstanceData, PixelData)):
        result = {}
        # meta信息
        if hasattr(obj, 'metainfo'):
            result['metainfo'] = dict(obj.metainfo)
        # data fields
        for k in obj._data_fields:
            v = getattr(obj, k)
            result[k] = datasample_to_dict(v)
        return result
    # tensor/ndarray
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj
    # list/tuple
    elif isinstance(obj, tuple):
        return tuple(datasample_to_dict(i) for i in obj)
    elif isinstance(obj, list):
        return [datasample_to_dict(i) for i in obj]
    # dict
    elif isinstance(obj, dict):
        return {k: datasample_to_dict(v) for k, v in obj.items()}
    # 基础类型
    else:
        return obj

def dict_to_datasample(d):
    datasample = DetDataSample()
    for k, v in d.items():
        # 处理 metainfo
        if k == 'metainfo':
            datasample.set_metainfo(v)
        # 处理 InstanceData 字段（带下划线和不带下划线都处理）
        elif k.lstrip('_') in ['gt_instances', 'pred_instances', 'proposals', 'ignored_instances', 'pred_track_instances']:
            setattr(datasample, k, dict_to_instancedata(v))
        # 处理 PixelData 字段
        elif k.lstrip('_') in ['gt_panoptic_seg', 'pred_panoptic_seg', 'gt_sem_seg', 'pred_sem_seg']:
            setattr(datasample, k, dict_to_pixeldata(v))
        else:
            setattr(datasample, k, v)
    return datasample

def dict_to_instancedata(d):
    """dict 递归还原为 InstanceData。"""
    inst = InstanceData()
    for k, v in d.items():
        if k == 'metainfo':
            inst.set_metainfo(v)
        else:
            setattr(inst, k, dict_to_datasample(v) if isinstance(v, dict) and not isinstance(v, (torch.Tensor, np.ndarray)) else v)
    return inst

def dict_to_pixeldata(d):
    """dict 递归还原为 PixelData。"""
    pix = PixelData()
    for k, v in d.items():
        if k == 'metainfo':
            pix.set_metainfo(v)
        else:
            setattr(pix, k, dict_to_datasample(v) if isinstance(v, dict) and not isinstance(v, (torch.Tensor, np.ndarray)) else v)
    return pix
# class CoopPromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         n_cls = len(classnames) #类别数量
#         n_ctx = cfg.TRAINER.COOP.N_CTX #提示长度
#         ctx_init = cfg.TRAINER.COOP.CTX_INIT #无，可以忽视
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         clip_imsize = clip_model.visual.input_resolution
#         cfg_imsize = cfg.INPUT.SIZE[0]
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

#         if ctx_init:#"xxxxxx"
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#             prompt = clip.tokenize(ctx_init)
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
#             #切片[1: 1 + n_ctx]选择了除起始token之外的n_ctx个token对应的嵌入向量
#             prompt_prefix = ctx_init

#         else:
#             # random initialization
#             if cfg.TRAINER.COOP.CSC:
#                 print("Initializing class-specific contexts")
#                 ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#             else:
#                 print("Initializing a generic context")
#                 ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)

#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")

#         self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]

#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens
#         self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

#     def forward(self):
#         ctx = self.ctx
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

#         prefix = self.token_prefix
#         suffix = self.token_suffix

#         if self.class_token_position == "end":
#             prompts = torch.cat(
#                 [
#                     prefix,  # (n_cls, 1, dim)
#                     ctx,     # (n_cls, n_ctx, dim)
#                     suffix,  # (n_cls, *, dim)
#                 ],
#                 dim=1,
#             )

#         elif self.class_token_position == "middle":
#             half_n_ctx = self.n_ctx // 2
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i : i + 1, :, :]
#                 class_i = suffix[i : i + 1, :name_len, :]
#                 suffix_i = suffix[i : i + 1, name_len:, :]
#                 ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
#                 ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,     # (1, 1, dim)
#                         ctx_i_half1,  # (1, n_ctx//2, dim)
#                         class_i,      # (1, name_len, dim)
#                         ctx_i_half2,  # (1, n_ctx//2, dim)
#                         suffix_i,     # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)

#         elif self.class_token_position == "front":
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i : i + 1, :, :]
#                 class_i = suffix[i : i + 1, :name_len, :]
#                 suffix_i = suffix[i : i + 1, name_len:, :]
#                 ctx_i = ctx[i : i + 1, :, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         class_i,   # (1, name_len, dim)
#                         ctx_i,     # (1, n_ctx, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)

#         else:
#             raise ValueError

#         return prompts


# class CocoopPromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = cfg.TRAINER.COCOOP.N_CTX
#         ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         vis_dim = clip_model.visual.output_dim
#         clip_imsize = clip_model.visual.input_resolution
#         cfg_imsize = cfg.INPUT.SIZE[0]
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

#         if ctx_init:
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#             prompt = clip.tokenize(ctx_init)
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
#             prompt_prefix = ctx_init
#         else:
#             # random initialization
#             ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)

#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")

#         self.ctx = nn.Parameter(ctx_vectors)

#         self.meta_net = nn.Sequential(OrderedDict([
#             ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
#             ("relu", nn.ReLU(inplace=True)),
#             ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
#         ]))

#         if cfg.TRAINER.COCOOP.PREC == "fp16":
#             self.meta_net.half()

#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]

#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens

#     def construct_prompts(self, ctx, prefix, suffix, label=None):
#         # dim0 is either batch_size (during training) or n_cls (during testing)
#         # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
#         # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
#         # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

#         if label is not None:
#             prefix = prefix[label]
#             suffix = suffix[label]

#         prompts = torch.cat(
#             [
#                 prefix,  # (dim0, 1, dim)
#                 ctx,  # (dim0, n_ctx, dim)
#                 suffix,  # (dim0, *, dim)
#             ],
#             dim=1,
#         )

#         return prompts

#     def forward(self, im_features):
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#         ctx = self.ctx  # (n_ctx, ctx_dim)
#         bias = self.meta_net(im_features)  # (batch, ctx_dim)
#         bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
#         ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
#         ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

#         # Use instance-conditioned context tokens for all classes
#         prompts = []
#         for ctx_shifted_i in ctx_shifted:
#             ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
#             pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
#             prompts.append(pts_i)
#         prompts = torch.stack(prompts)

#         return prompts


# class MultiModalPromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = cfg.TRAINER.MAPLE.N_CTX
#         ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         clip_imsize = clip_model.visual.input_resolution
#         cfg_imsize = cfg.INPUT.SIZE[0]
#         # Default is 1, which is compound shallow prompting
#         assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
#         self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

#         if ctx_init and (n_ctx) <= 4:
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = n_ctx
#             prompt = clip.tokenize(ctx_init)
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
#             prompt_prefix = ctx_init
#         else:
#             # random initialization
#             ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)
#         print('MaPLe design: Multi-modal Prompt Learning')
#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of MaPLe context words (tokens): {n_ctx}")
#         # These below, related to the shallow prompts
#         # Linear layer so that the tokens will project to 512 and will be initialized from 768
#         self.proj = nn.Linear(ctx_dim, 768)
#         self.proj.half()
#         self.ctx = nn.Parameter(ctx_vectors)
#         # These below parameters related to the shared prompts
#         # Define the compound prompts for the deeper layers

#         # Minimum can be 1, which defaults to shallow MaPLe
#         # compound prompts
#         self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
#                                                       for _ in range(self.compound_prompts_depth - 1)])
#         for single_para in self.compound_prompts_text:
#             nn.init.normal_(single_para, std=0.02)
#         # Also make corresponding projection layers, for each prompt
#         single_layer = nn.Linear(ctx_dim, 768)
#         self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]

#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens

#     def construct_prompts(self, ctx, prefix, suffix, label=None):
#         # dim0 is either batch_size (during training) or n_cls (during testing)
#         # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
#         # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
#         # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

#         if label is not None:
#             prefix = prefix[label]
#             suffix = suffix[label]

#         prompts = torch.cat(
#             [
#                 prefix,  # (dim0, 1, dim)
#                 ctx,  # (dim0, n_ctx, dim)
#                 suffix,  # (dim0, *, dim)
#             ],
#             dim=1,
#         )

#         return prompts

#     def forward(self):
#         ctx = self.ctx

#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

#         prefix = self.token_prefix
#         suffix = self.token_suffix
#         prompts = self.construct_prompts(ctx, prefix, suffix)

#         # Before returning, need to transform
#         # prompts to 768 for the visual side
#         visual_deep_prompts = []
#         for index, layer in enumerate(self.compound_prompt_projections):
#             visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
#         # Now the other way around
#         # We will project the textual prompts from 512 to 768
#         return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required

##implement from MRGDINO

# from timm.models.layers import trunc_normal_

# lan_scale = 0.1
# vis_scale = 0.1



# class RepZeroLinear(nn.Linear):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
#         super().__init__(in_features, out_features, bias, device, dtype)
#         self.scaling = nn.parameter.Parameter(torch.ones(1) * lan_scale)
#         nn.init.constant_(self.weight, val=zero_value)
#         self.freeze_linear = nn.Linear(in_features, out_features, bias, device, dtype)
#         nn.init.constant_(self.freeze_linear.weight, val=0.0)
#         if self.bias is not None:
#             nn.init.constant_(self.freeze_linear.bias, val=0.0) 
        
#         # self.zero_inter_loss = torch.nn.L1Loss(reduction='mean')
#         # self.zero_inter_loss = torch.nn.MSELoss(reduction='mean')
#         self.zero_inter_loss = torch.nn.SmoothL1Loss(reduction='mean')

#     def forward(self, input: Tensor) -> Tensor:
#         if self.training:
#             branch_output = self.scaling * super().forward(input)
#             output = branch_output + self.freeze_linear(input)
#             return output, \
#                 self.zero_inter_loss(branch_output, torch.zeros_like(branch_output)) + \
#                     self.zero_inter_loss(output, torch.zeros_like(output))
#         else:
#             return self.freeze_linear(input), torch.zeros(1).to(input)

#     def __rep__(self):
#         self.freeze_linear.weight.data = self.weight.data  * self.scaling + self.freeze_linear.weight.data
#         self.freeze_linear.bias.data = self.bias.data  * self.scaling + self.freeze_linear.bias.data
#         self.scaling = nn.parameter.Parameter(torch.ones(1).to(self.weight.data) * lan_scale)
#         nn.init.constant_(self.weight, val=zero_value)
#         if self.bias is not None:
#             nn.init.constant_(self.bias, val=zero_value)
            
            


# def shift_columns(tensor, m, k):
#     """
#     将前 m 列整体向右移动 k 列，并保持剩下的列为 -inf
#     参数:
#     - tensor: 需要操作的二维 tensor
#     - m: 前 m 列是非-inf 的部分
#     - k: 向右移动的列数
#     """
#     # 获取 tensor 的行数和列数
#     num_rows, num_cols = tensor.shape

#     # 创建新的 tensor，初始化为 -inf
#     new_tensor = torch.full((num_rows, num_cols), float('-inf'))

#     # 计算移动后有效列的起始索引和终止索引
#     start_idx = k
#     end_idx = min(k + m, num_cols)  # 确保不会超出边界

#     # 将原 tensor 的前 m 列复制到新的 tensor 的 [k:k+m] 位置
#     new_tensor[:, start_idx:end_idx] = tensor[:, :end_idx - k]

#     return new_tensor


# def find_inf_boundary(t):
#     # 创建一个布尔掩码，检查每个元素是否为 -inf
#     is_inf = t == float('-inf')
    
#     # 对每一列求和，看看是否整列都为 -inf
#     # 使用 all(dim=0) 来检查每一列是否都是 True (-inf)
#     inf_cols = is_inf.all(dim=0)
    
#     # 返回第一个全为 -inf 的列的索引
#     inf_boundary_index = torch.nonzero(inf_cols).min().item()
    
#     return inf_boundary_index


# class CoOpModule(nn.Module):
#     def __init__(self, prompt_length, prompt_channel, use_prompt=False, prompt=None) -> None:
#         super().__init__()
#         self.prompt_length = prompt_length
#         self.prompt_channel = prompt_channel
#         if use_prompt:
#             self.coop_prompt = prompt
#         else:
#             self.coop_prompt = nn.Parameter(torch.zeros(1, self.prompt_length, self.prompt_channel))
#             trunc_normal_(self.coop_prompt, std=0.02)
    
#     def forward(self, x):
#         return x



# class Prompt(nn.Module):
#     def __init__(self, length=4, embed_dim=768, embed_dim_key=768, embedding_key='mean', prompt_init='uniform', prompt_pool=True, 
#                  prompt_key=True, pool_size=10, top_k=4, batchwise_prompt=True, prompt_key_init='uniform',):
#         super().__init__()

#         self.length = length
#         self.embed_dim = embed_dim
#         self.embed_dim_key = embed_dim_key
#         self.prompt_pool = prompt_pool
#         self.embedding_key = embedding_key
#         self.prompt_init = prompt_init
#         self.prompt_key = prompt_key
#         self.pool_size = pool_size
#         self.top_k = top_k
#         self.batchwise_prompt = batchwise_prompt

#         if self.prompt_pool:
#             prompt_pool_shape = (pool_size, length, embed_dim)
#             if prompt_init == 'zero':
#                 self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#             elif prompt_init == 'uniform':
#                 self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
#                 nn.init.uniform_(self.prompt, -1, 1)
        
#         # if using learnable prompt keys
#         if prompt_key:
#             key_shape = (pool_size, embed_dim_key)
#             if prompt_key_init == 'zero':
#                 self.prompt_key = nn.Parameter(torch.zeros(key_shape))
#             elif prompt_key_init == 'uniform':
#                 self.prompt_key = nn.Parameter(torch.randn(key_shape))
#                 nn.init.uniform_(self.prompt_key, -1, 1)
#         else:
#             # else use mean of prompt as key
#             # only compatible with prompt, not prefix
#             prompt_mean = torch.mean(self.prompt, dim=1)
#             self.prompt_key = prompt_mean
    
#     def l2_normalize(self, x, dim=None, epsilon=1e-12):
#         """Normalizes a given vector or matrix."""
#         square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
#         x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
#         return x * x_inv_norm
    
#     def forward(self, x_embed, prompt_mask=None, cls_features=None):
#         out = dict()
#         if self.prompt_pool:
#             if self.embedding_key == 'mean':
#                 x_embed_mean = torch.mean(x_embed, dim=1)
#             elif self.embedding_key == 'max':
#                 x_embed_mean = torch.max(x_embed, dim=1)[0]
#             elif self.embedding_key == 'mean_max':
#                 x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
#             elif self.embedding_key == 'cls':
#                 if cls_features is None:
#                     x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
#                 else:
#                     x_embed_mean = cls_features
#             else:
#                 raise NotImplementedError("Not supported way of calculating embedding keys!")

#             prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
#             x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

#             similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
#             if prompt_mask is None:
#                 _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
#                 if self.batchwise_prompt:
#                     prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
#                     # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
#                     # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
#                     # Unless dimension is specified, this will be flattend if it is not already 1D.
#                     if prompt_id.shape[0] < self.pool_size:
#                         prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
#                         id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
#                     _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
#                     major_prompt_id = prompt_id[major_idx] # top_k
#                     # expand to batch
#                     idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
#             else:
#                 idx = prompt_mask # B, top_k

#             batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
#             batch_size, top_k, length, c = batched_prompt_raw.shape
#             batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

#             # out['prompt_idx'] = idx

#             # # Debugging, return sim as well
#             # out['prompt_norm'] = prompt_norm
#             # out['x_embed_norm'] = x_embed_norm
#             # out['similarity'] = similarity

#             # # Put pull_constraint loss calculation inside
#             # batched_key_norm = prompt_norm[idx] # B, top_k, C
#             # out['selected_key'] = batched_key_norm
#             # x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
#             # sim = batched_key_norm * x_embed_norm # B, top_k, C
#             # reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

#             # out['reduce_sim'] = reduce_sim

        
#         # The input with the prompt concatenated to the front. [B, prompt+token, C]
#         # out['total_prompt_len'] = batched_prompt.shape[1]
#         # out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

#         return batched_prompt

# zero_value = 1e-8
# class RepZeroConv2d(nn.Conv2d):
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding = 0,
#                  dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, zero_value=zero_value) -> None:
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
#         self.scaling = nn.parameter.Parameter(torch.ones(1) * vis_scale)
#         nn.init.constant_(self.weight, val=zero_value)
#         if self.bias is not None:
#             nn.init.constant_(self.bias, val=zero_value)
        
#         self.freeze_conv = nn.Conv2d(in_channels, out_channels,
#                                      kernel_size, stride, padding,
#                                      dilation, groups, bias,
#                                      padding_mode, device, dtype)
#         nn.init.constant_(self.freeze_conv.weight, val=0.0)
#         if self.bias is not None:
#             nn.init.constant_(self.freeze_conv.bias, val=0.0)
        
#         # self.zero_inter_loss = torch.nn.L1Loss(reduction='mean')
#         # self.zero_inter_loss = torch.nn.MSELoss(reduction='mean')
#         self.zero_inter_loss = torch.nn.SmoothL1Loss(reduction='mean')

#     def forward(self, input: Tensor) -> Tensor:
#         if self.training:
#             branch_output = self.scaling * super().forward(input)
#             output = branch_output + self.freeze_conv(input)
#             return output, \
#                 self.zero_inter_loss(branch_output, torch.zeros_like(branch_output)) + \
#                     self.zero_inter_loss(output, torch.zeros_like(output))
#         else:
#             return self.freeze_conv(input), torch.zeros(1).to(input)

#     def __rep__(self):
#         self.freeze_conv.weight.data = self.weight.data  * self.scaling + self.freeze_conv.weight.data
#         self.freeze_conv.bias.data = self.bias.data  * self.scaling + self.freeze_conv.bias.data
#         self.scaling = nn.parameter.Parameter(torch.ones(1).to(self.weight.data) *vis_scale)
#         nn.init.constant_(self.weight, val=zero_value)
#         if self.bias is not None:
#             nn.init.constant_(self.bias, val=zero_value)