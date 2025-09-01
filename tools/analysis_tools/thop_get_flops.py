# Copyright (c) OpenMMLab. All rights reserved.
# Modified to use thop for FLOPs calculation.
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version
from tqdm import tqdm

from mmdet.registry import MODELS

# 关键改动：导入 thop
try:
    from thop import profile
except ImportError:
    raise ImportError(
        'Please install thop to use this script. `pip install thop`'
    )

import inspect
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData,PixelData
import torch
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

def smart_compare(a, b, path="root"):
    """递归比较两个对象/字典/列表/张量是否内容一致，并输出详细差异。"""
    if type(a) != type(b):
        print(f"[类型不一致] {path}: {type(a)} != {type(b)}")
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            print(f"[dict键不一致] {path}: {set(a.keys())} != {set(b.keys())}")
            return False
        for k in a:
            if not smart_compare(a[k], b[k], path + f".{k}"):
                return False
        return True
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            print(f"[list长度不一致] {path}: {len(a)} != {len(b)}")
            return False
        for i, (x, y) in enumerate(zip(a, b)):
            if not smart_compare(x, y, path + f"[{i}]"):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            print(f"[Tensor不一致] {path}:\n{a}\n!=\n{b}")
            return False
        return True
    elif hasattr(a, '__dict__') and hasattr(b, '__dict__'):
        return smart_compare(vars(a), vars(b), path)
    else:
        if a != b:
            print(f"[值不一致] {path}: {a} != {b}")
            return False
        return True

def get_model_forward_args(model):
    """
    使用 inspect 模块自动获取模型前向传播函数所需的额外参数列表。

    Args:
        model (torch.nn.Module): 需要分析的模型。

    Returns:
        list: 一个包含额外参数名称的字符串列表。
    """
    # 1. 确定要分析的目标函数 (优先使用 forward_dummy)
    # target_func = getattr(model, 'forward_dummy', model.forward)
    target_func = model.forward
    # 2. 获取函数签名
    try:
        sig = inspect.signature(target_func)
    except ValueError:
        # 如果是内置函数或某些C扩展，可能会失败
        print("无法获取函数签名。")
        return []

    # 3. 分析参数
    arg_names = []
    params_to_check = list(sig.parameters.values())[1:] 

    for param in params_to_check:
        # 只关心位置或关键字参数以及纯关键字参数
        if param.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                          inspect.Parameter.KEYWORD_ONLY]:
            arg_names.append(param.name)
            
    return arg_names

def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='num images of calculate model flops'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.'
    )
    args = parser.parse_args()
    return args

def inference(args, logger):
    """Main function to calculate FLOPs and parameters."""
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'PyTorch version < 1.12 might have compatibility issues.'
        )

    # --- 1. 配置和模型加载 (与原版基本相同) ---
    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1  # 必须是1，逐张图片计算
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model)

    # --- 2. 模型准备 (与原版相同) ---
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 将 SyncBatchNorm 转换为普通 BatchNorm，这对于单卡推理/分析至关重要
    model = revert_sync_batchnorm(model)
    model.eval()

    # extra_args = get_model_forward_args(model)
    # if extra_args:
    #     logger.info(f"模型 forward 函数需要的参数: {extra_args}")
    # else:
    #     logger.info("模型 forward 函数似乎不需要额外的参数。")
    # import sys
    # sys.exit()
    # --- 3. 核心计算逻辑 (使用 thop) ---
    avg_flops_list = []
    total_params = 0
    last_ori_shape, last_pad_shape = None, None

    # 使用 tqdm 创建进度条
    for idx, data_batch in tqdm(enumerate(data_loader), total=args.num_images):
        if idx >= args.num_images:
            break

        # 数据预处理
        data = model.data_preprocessor(data_batch)
        inputs_tensor = data['inputs']
        data_samples = data['data_samples']

        # 记录最后一张图片的形状信息用于打印
        last_ori_shape = data_samples[0].ori_shape
        last_pad_shape = getattr(data_samples[0], 'batch_input_shape', data_samples[0].pad_shape)
        dict_list = data_samples_list_to_dict_list(data['data_samples'])
        # 使用 thop.profile 计算 FLOPs 和参数量
        # inputs 必须是元祖

        flops, params = profile(
            model,
            inputs=(inputs_tensor,),
            data_samples_dict_list=dict_list,
            verbose=False  # 关闭详细输出
        )
        
        avg_flops_list.append(flops)
        # 参数量是固定的，只在第一次循环时获取
        if total_params == 0:
            total_params = params

    del data_loader

    # --- 4. 结果格式化和返回 ---
    result = {}
    if not avg_flops_list:
        logger.warning('No images were processed. FLOPs calculation was skipped.')
        result['flops'] = '0.0 GFLOPs'
        result['params'] = '0.0 M'
        result['ori_shape'] = 'N/A'
        result['pad_shape'] = 'N/A'
        return result

    # 计算平均 FLOPs 并格式化
    mean_flops = np.average(avg_flops_list)
    result['flops'] = f'{mean_flops / 1e9:.2f} GFLOPs'  # 转换为 GFLOPs
    result['params'] = f'{total_params / 1e6:.2f} M'  # 转换为 M (百万)
    result['ori_shape'] = last_ori_shape
    result['pad_shape'] = last_pad_shape

    return result

def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    
    try:
        result = inference(args, logger)
        
        split_line = '=' * 30
        ori_shape = result['ori_shape']
        pad_shape = result['pad_shape']
        flops = result['flops']
        params = result['params']

        print(f'\n{split_line}')
        if pad_shape != ori_shape:
            print(f'Input shape was padded from {ori_shape} to {pad_shape}')
        
        print(f'Input shape: {pad_shape}')
        print(f'FLOPs: {flops}')
        print(f'Params: {params}')
        print(split_line)
        print('!!! Please be cautious: Results are based on thop. Check if all ops are supported.')
        print('!!! MMDetection models were assumed to accept `mode=\'tensor\'`.')

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()