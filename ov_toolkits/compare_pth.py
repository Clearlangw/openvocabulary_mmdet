import torch
import numpy as np
from pathlib import Path

def compare_pth_files(file1_path, file2_path):
    """比较两个pth文件并检查它们是否完全相同。
    
    参数:
        file1_path (str): 第一个pth文件的路径
        file2_path (str): 第二个pth文件的路径
    """
    print(f"正在比较文件：\n{file1_path}\n{file2_path}")
    print("-" * 50)
    
    # 加载pth文件
    tensor1 = torch.load(file1_path, map_location='cpu')
    tensor2 = torch.load(file2_path, map_location='cpu')
    
    # 检查是否为张量
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        print("错误：文件内容不是张量！")
        return False
    
    # 检查形状是否相同
    if tensor1.shape != tensor2.shape:
        print("张量形状不匹配：")
        print(f"文件1的形状: {tensor1.shape}")
        print(f"文件2的形状: {tensor2.shape}")
        return False
    
    # 检查张量是否完全相同
    if torch.equal(tensor1, tensor2):
        print("两个张量完全相同！")
        return True
    else:
        print("两个张量不同")
        # 计算差异统计信息
        diff = torch.abs(tensor1 - tensor2)
        print(f"最大差异: {diff.max().item()}")
        print(f"平均差异: {diff.mean().item()}")
        print(f"标准差: {diff.std().item()}")
        return False

if __name__ == "__main__":
    # 使用示例
    file1_path = "/home/wuke_2024/ov202503/mmdetection/ov_text_feature/background_feature.pth"
    file2_path = "/home/wuke_2024/ov202503/mmdetection/ov_text_feature/dino_bert_background_feature.pth"
    
    # 执行比较
    compare_pth_files(file1_path, file2_path)