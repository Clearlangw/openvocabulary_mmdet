# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from ..losses import QualityFocalLoss
from ..utils import multi_apply
from .deformable_detr_head import DeformableDETRHead

# @MODELS.register_module()
# class MIMHead():