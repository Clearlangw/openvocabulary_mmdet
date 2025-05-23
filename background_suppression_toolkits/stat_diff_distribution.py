import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_score_distributions(zero_shot_bg, zero_shot_ori, full_train_bg, full_train_ori):
    """
    分析并比较两种情况下分数的分布差异
    
    Args:
        zero_shot_bg: zeroshot下的background_similarity
        zero_shot_ori: zeroshot下的original_scores
        full_train_bg: full training下的background_similarity
        full_train_ori: full training下的original_scores
    """
    # 1. 基本统计量比较
    print("=== 基本统计量比较 ===")
    metrics = {
        'background_similarity (zeroshot)': zero_shot_bg,
        'background_similarity (full train)': full_train_bg,
        'original_scores (zeroshot)': zero_shot_ori,
        'original_scores (full train)': full_train_ori
    }
    
    for name, data in metrics.items():
        print(f"\n{name}:")
        print(f"  最小值: {np.min(data):.4f}")
        print(f"  最大值: {np.max(data):.4f}")
        print(f"  均值: {np.mean(data):.4f}")
        print(f"  中位数: {np.median(data):.4f}")
        print(f"  标准差: {np.std(data):.4f}")
        print(f"  25%分位数: {np.percentile(data, 25):.4f}")
        print(f"  75%分位数: {np.percentile(data, 75):.4f}")
    
    # 2. 分布差异分析
    print("\n=== 分布差异分析 ===")
    # 使用KS检验比较分布差异
    ks_bg = stats.ks_2samp(zero_shot_bg, full_train_bg)
    ks_ori = stats.ks_2samp(zero_shot_ori, full_train_ori)
    print(f"Background Similarity KS检验 p值: {ks_bg.pvalue:.4f}")
    print(f"Original Scores KS检验 p值: {ks_ori.pvalue:.4f}")
    
    # 3. 长尾分析 (original_scores > -1)
    print("\n=== 长尾分析 (original_scores > -1) ===")
    zero_shot_tail = zero_shot_ori[zero_shot_ori > -1]
    full_train_tail = full_train_ori[full_train_ori > -1]
    
    print(f"Zeroshot长尾样本数: {len(zero_shot_tail)}")
    print(f"Full train长尾样本数: {len(full_train_tail)}")
    
    if len(zero_shot_tail) > 0 and len(full_train_tail) > 0:
        print("\n长尾部分统计量:")
        print("Zeroshot长尾:")
        print(f"  均值: {np.mean(zero_shot_tail):.4f}")
        print(f"  中位数: {np.median(zero_shot_tail):.4f}")
        print(f"  标准差: {np.std(zero_shot_tail):.4f}")
        
        print("\nFull train长尾:")
        print(f"  均值: {np.mean(full_train_tail):.4f}")
        print(f"  中位数: {np.median(full_train_tail):.4f}")
        print(f"  标准差: {np.std(full_train_tail):.4f}")
        
        # 长尾部分的KS检验
        ks_tail = stats.ks_2samp(zero_shot_tail, full_train_tail)
        print(f"\n长尾部分KS检验 p值: {ks_tail.pvalue:.4f}")
    
    # 4. 可视化比较
    plt.figure(figsize=(15, 10))
    
    # 4.1 原始分数分布对比
    plt.subplot(2, 2, 1)
    plt.hist(zero_shot_ori, bins=50, alpha=0.5, label='Zeroshot', density=True)
    plt.hist(full_train_ori, bins=50, alpha=0.5, label='Full train', density=True)
    plt.title('Original Scores Distribution')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    
    # 4.2 背景相似度分布对比
    plt.subplot(2, 2, 2)
    plt.hist(zero_shot_bg, bins=50, alpha=0.5, label='Zeroshot', density=True)
    plt.hist(full_train_bg, bins=50, alpha=0.5, label='Full train', density=True)
    plt.title('Background Similarity Distribution')
    plt.xlabel('Similarity')
    plt.ylabel('Density')
    plt.legend()
    
    # 4.3 长尾部分分布对比
    plt.subplot(2, 2, 3)
    if len(zero_shot_tail) > 0:
        plt.hist(zero_shot_tail, bins=30, alpha=0.5, label='Zeroshot', density=True)
    if len(full_train_tail) > 0:
        plt.hist(full_train_tail, bins=30, alpha=0.5, label='Full train', density=True)
    plt.title('Long Tail Distribution (Score > -1)')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    
    # 4.4 箱线图比较
    plt.subplot(2, 2, 4)
    data = [zero_shot_ori, full_train_ori, zero_shot_bg, full_train_bg]
    labels = ['Zeroshot\nOriginal', 'Full train\nOriginal', 
              'Zeroshot\nBackground', 'Full train\nBackground']
    plt.boxplot(data, tick_labels=labels)
    plt.title('Box Plot Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('score_distribution_analysis.png')
    plt.close()

# 使用示例
if __name__ == "__main__":
    # 加载数据
    zero_shot_bg = np.load('/home/wuke_2024/ov202503/mmdetection/zero_shot_background_similarity.npy').flatten()
    zero_shot_ori = np.load('/home/wuke_2024/ov202503/mmdetection/zero_shot_original_scores.npy').flatten()
    full_train_bg = np.load('/home/wuke_2024/ov202503/mmdetection/full_training_background_similarity.npy').flatten()
    full_train_ori = np.load('/home/wuke_2024/ov202503/mmdetection/full_training_original_scores.npy').flatten()
    
    # 运行分析
    analyze_score_distributions(zero_shot_bg, zero_shot_ori, full_train_bg, full_train_ori)