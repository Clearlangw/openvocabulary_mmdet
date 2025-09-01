# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version

from mmdet.registry import MODELS
from tqdm import tqdm
try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')

from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData,PixelData
import torch
import time
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

def unit_test_datasample_to_dict_and_back():

    sample = DetDataSample()
    sample.ori_shape = (480, 640, 3)
    sample.pad_shape = (512, 640, 3)
    sample.img_shape = (512, 640, 3)
    sample.set_metainfo({'filename': 'test.jpg'})
    sample.gt_instances = InstanceData(bboxes=torch.tensor([[1, 2, 3, 4]]))
    sample.pred_instances = InstanceData(bboxes=torch.tensor([[5, 6, 7, 8]]))
    d = datasample_to_dict(sample)
    sample2 = dict_to_datasample(d)

    assert smart_compare(sample, sample2)
    print("unit test datasample_to_dict_and_back passed")

def example_test_datasample_to_dict_and_back(sample):

    d = datasample_to_dict(sample)
    sample2 = dict_to_datasample(d)

    assert smart_compare(sample, sample2)
    print("example test datasample_to_dict_and_back passed")



def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='num images of calculate model flops')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--param-keyword',
        type=str,
        default='',
        help='If set, count the number of parameters whose name contains this keyword.'
    )
    args = parser.parse_args()
    return args


def inference(args, logger):
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment，
    # this question involves models like configs/strong_baselines
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)

    result = {}
    avg_flops = []
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model)
    # print(model.summary())
    # print(model.__dict__)
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.numel()}")
    
    # total = 0
    # for name, param in model.named_parameters():
    #     if 'language_model' not in name:
    #         #print(f"{name}: {param.numel()}")
    #         total += param.numel()
    # print(f"Total parameters (excluding 'language_model'): {total}")

    # total = 0
    # for name, param in model.named_parameters():
    #     if 'backbone' in name and 'language_model' not in name:
    #         #print(f"{name}: {param.numel()}")
    #         total += param.numel()
    # print(f"Total parameters (only 'backbone'): {total}")

    # total = 0
    # for name, param in model.named_parameters():
    #     if 'language_model' in name:
    #         #print(f"{name}: {param.numel()}")
    #         total += param.numel()
    # print(f"Total parameters (only 'language_model'): {total}")

    # total = 0
    # for name, param in model.named_parameters():
    #     total += param.numel()
    # print(f"Total named parameters: {total}")

    
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    if hasattr(args, 'param_keyword') and args.param_keyword:
        num = count_parameters_by_name(model, args.param_keyword)
        result['param_count'] = (args.param_keyword, num)
        print("?"*100)
        print(f"Total parameters (keyword '{args.param_keyword}'): {num}")
        print("?"*100)
    # 新增统计可训练参数量
    trainable_num = count_trainable_parameters(model)
    print("!"*100)
    print(f"Trainable parameters: {trainable_num}")
    print("!"*100)

    if hasattr(model, 'forward_dummy'):
        _forward = model.forward_dummy
    else:
        _forward = model.forward

    # === 统计FPS相关 ===
    start_time = time.time()
    img_count = 0

    for idx, data_batch in tqdm(enumerate(data_loader)):
        if idx == args.num_images:
            break
        data = model.data_preprocessor(data_batch)
        result['ori_shape'] = data['data_samples'][0].ori_shape
        result['pad_shape'] = data['data_samples'][0].pad_shape
        if hasattr(data['data_samples'][0], 'batch_input_shape'):
            result['pad_shape'] = data['data_samples'][0].batch_input_shape
        
        dict_list = data_samples_list_to_dict_list(data['data_samples'])
        #data_samples = dict_list_to_data_samples_list(dict_list)
        #原本：model.forward = partial(_forward, data_samples=data['data_samples'])
        model.forward = partial(_forward, data_samples_dict_list=dict_list)
        outputs = get_model_complexity_info(
            model,
            None,
            inputs=data['inputs'],
            show_table=False,
            show_arch=False)
        avg_flops.append(outputs['flops'])
        params = outputs['params']
        result['compute_type'] = 'dataloader: load a picture from the dataset'
        img_count += 1

    end_time = time.time()
    del data_loader

    # === 计算FPS ===
    total_time = end_time - start_time
    fps = img_count / total_time * 1000 if total_time > 0 else 0
    result['fps'] = fps

    mean_flops = _format_size(int(np.average(avg_flops)))
    params = _format_size(params)
    result['flops'] = mean_flops
    result['params'] = params

    return result

def count_parameters_by_name(model, keyword):
    """统计模型中参数名包含keyword的参数总数。"""
    total = 0
    for name, param in model.named_parameters():
        if keyword in name:
            total += param.numel()
    return total

def count_trainable_parameters(model):
    """统计模型中所有可训练参数（requires_grad=True）且为语言模型和backbone的总数。"""
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'language_model' in name or 'backbone' in name:
                # print(f"name: {name}  is trainable")
                # import sys 
                # sys.exit()
                total += param.numel()
    return total


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']

    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')
    print(f'{split_line}\nCompute type: {compute_type}\n'
          f'Input shape: {pad_shape}\nFlops: {flops}\n'
          f'Params: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')

    # 新增：如果设置了param-keyword，则统计参数量
    if 'param_count' in result:
        keyword, num = result['param_count']
        print(f"Total parameters (keyword '{keyword}'): {num}")

    if 'fps' in result:
        print(f"Average FPS (images per second): {result['fps']:.2f}")


if __name__ == '__main__':
    main()
