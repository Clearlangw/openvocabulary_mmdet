import numpy as np
import matplotlib.pyplot as plt

# 加载数据
background_similarity = np.load('/home/wuke_2024/ov202503/mmdetection/right_zero_shot_background_similarity.npy').flatten()
original_scores = np.load('/home/wuke_2024/ov202503/mmdetection/right_zero_shot_original_scores.npy').flatten()

plt.figure(figsize=(10, 5))

# 绘制背景相似度分布
plt.subplot(1, 2, 1)
plt.hist(background_similarity, bins=50, color='skyblue', alpha=0.7)
plt.title('Background Similarity Distribution')
plt.xlabel('background_similarity')
plt.ylabel('Count')

# 绘制原始分数分布
plt.subplot(1, 2, 2)
plt.hist(original_scores, bins=50, color='salmon', alpha=0.7)
plt.title('Original Scores Distribution')
plt.xlabel('original_scores')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('right_zero_shot_score_distribution.png')  # 保存为图片