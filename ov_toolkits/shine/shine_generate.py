import os
import json
import torch
import random
from transformers import BertTokenizer, BertModel
from typing import List, Dict

# ====== 1. 配置 ======
# 包含句子的JSON输入文件
INPUT_JSON_FILE = "generated_sentences_by_class.json"

# 保存最终特征向量的目录
OUTPUT_FEATURE_DIR = "/home/wuke_2024/ov202503/mmdetection/ov_text_feature/shine_class_features"

# BERT模型路径或名称
LANG_MODEL_PATH = '/home/wuke_2024/ov202503/text_encoder/bert-base-uncased'

# --- 关键配置: 定义下游模型最终使用的类别顺序 ---
# DETECTION_PROMPT_CLASSES = [
#     "pedestrian XX XX", "people XX XX", "bicycle XX XX", "car XX XX", "van XX XX",
#     "truck XX XX", "tricycle XX", "awning-tricycle", "bus XX XX", "motor XX XX"
# ]

# --- 其他配置 ---
NUM_BERT_LAYERS_TO_AVG = 1
NUM_OUTPUT_TOKENS = 3
DROPOUT_RATE_RANGE = (0.3, 0.5) 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ========================

def save_embedding(emb: torch.Tensor, save_dir: str, filename: str):
    """保存单个嵌入张量到文件。"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(emb, path)
    print(f"  -> 已保存纯语义特征至: {path}")

def get_batch_cls_embeddings(
    texts: List[str],
    tokenizer,
    model,
    num_layers: int,
    device: str
) -> torch.Tensor:
    """为一批句子生成[CLS] token的纯语义嵌入。"""
    if not texts:
        return torch.empty(0, device=device)

    inputs = tokenizer(
        texts, return_tensors="pt", padding="longest", truncation=True, max_length=256
    ).to(device)
    # print(inputs['input_ids'].shape)
    # print(inputs)
    # import sys
    # sys.exit()

    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_hidden_states=True
        )

    encoded_layers = outputs.hidden_states[1:]
    features_avg = torch.stack(encoded_layers[-num_layers:], dim=0).mean(dim=0)
    cls_embeddings = features_avg[:, 0, :]
    
    return cls_embeddings

def generate_features():
    """完整的主流程函数。"""
    print("--- 开始生成纯语义特征 (无位置信息) ---")
    print(f"使用设备: {DEVICE}")

    # 1. 加载句子数据
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            sentences_by_class = json.load(f)
        print(f"成功加载 {len(sentences_by_class)} 个类别的句子数据。")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {INPUT_JSON_FILE}")
        return

    # 2. 初始化BERT模型和分词器
    print(f"正在加载模型: {LANG_MODEL_PATH} ...")
    try:
        tokenizer = BertTokenizer.from_pretrained(LANG_MODEL_PATH)
        model = BertModel.from_pretrained(LANG_MODEL_PATH).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        return
    
    # 3. 逐类处理并生成特征
    print("\n--- 开始处理每个类别 ---")
    for category, sentences in sentences_by_class.items():
        if not sentences:
            print(f"类别 '{category}' 没有句子，已跳过。")
            continue
        
        print(f"正在处理类别: '{category}' (包含 {len(sentences)} 条句子)...")

        try:
            # 3a. 获取该类别所有句子的[CLS]嵌入
            batch_cls_embeddings = get_batch_cls_embeddings(
                texts=sentences,
                tokenizer=tokenizer,
                model=model,
                num_layers=NUM_BERT_LAYERS_TO_AVG,
                device=DEVICE
            )

            if batch_cls_embeddings.shape[0] == 0:
                print(f"  -> 未能为类别 '{category}' 生成任何嵌入。")
                continue

            # 3b. 聚合生成多Token语义特征
            num_sentences = batch_cls_embeddings.shape[0]
            semantic_tokens = []

            # Token 1: 全局视图
            semantic_tokens.append(torch.mean(batch_cls_embeddings, dim=0))
            
            # Token 2, 3, ...: 随机视图
            for i in range(1, NUM_OUTPUT_TOKENS):
                dropout_rate = random.uniform(DROPOUT_RATE_RANGE[0], DROPOUT_RATE_RANGE[1])
                num_to_keep = max(1, int(num_sentences * (1 - dropout_rate)))
                indices = torch.randperm(num_sentences, device=DEVICE)[:num_to_keep]
                semantic_tokens.append(torch.mean(batch_cls_embeddings[indices], dim=0))
            
            # 3c. 将所有Token堆叠成最终的纯语义特征张量
            pure_semantic_feature = torch.stack(semantic_tokens, dim=0).unsqueeze(0) # -> (1, 3, 768)
            print(f"  -> 生成的纯语义特征形状: {pure_semantic_feature.shape}")

            # 3d. 直接保存这个纯语义特征
            output_filename = f"{category}_feature.pth"
            save_embedding(pure_semantic_feature, OUTPUT_FEATURE_DIR, output_filename)

        except Exception as e:
            print(f"  -> 处理类别 '{category}' 时发生错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- 所有类别处理完毕 ---")


if __name__ == "__main__":
    generate_features()