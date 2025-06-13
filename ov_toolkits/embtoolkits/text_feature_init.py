import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from typing import Optional

def get_bert_input_embedding(
    text: str,
    remove_cls_sep: bool = True, # 参数名与函数定义一致
    device: str = 'cuda:0', # 为常用设置调整了默认设备
    lang_model_name: str = 'bert-base-uncased', # 简化了默认路径
    num_layers_of_embedded: int = 1 # 您“常规单词”处理中平均的BERT层数
) -> torch.Tensor:
    """
    为背景 token 生成 BERT 嵌入。
    特征提取方式与“常规单词”类似（深度上下文），
    但是点号 token 以及 [CLS]、[SEP] token 会被直接从序列中移除，
    而不是仅仅被清零。
    """
    tokenizer = BertTokenizer.from_pretrained(lang_model_name)
    model = BertModel.from_pretrained(lang_model_name).to(device)
    model.eval()

    # 使用 padding="longest" 以处理不同长度文本（如果未来需要批量处理，尽管当前函数针对bs=1）
    inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True).to(device)
    input_ids = inputs['input_ids']  # 形状: (batch_size, seq_len_padded)
    attention_mask_bert = inputs['attention_mask'] # 形状: (batch_size, seq_len_padded)

    if input_ids.size(0) != 1:
        # 当前实现中的选择和压缩逻辑假设 batch_size = 1
        # 原始函数的输出 view(1, -1, emb.size(-1)) 也暗示了输出是 batch_size 1
        raise ValueError("此函数当前期望输入文本的 batch_size = 1。")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask_bert,
            output_hidden_states=True
        )

    # outputs.hidden_states 是一个元组 (input_embeds + 对于基础模型是12个隐藏层输出)
    encoded_layers = outputs.hidden_states[1:] # 所有Transformer隐藏层

    if len(encoded_layers) < num_layers_of_embedded:
        raise ValueError(
            f"BERT 模型有 {len(encoded_layers)} 个隐藏层, "
            f"但 num_layers_of_embedded 设置为 {num_layers_of_embedded}。"
            "请确保 num_layers_of_embedded 不大于可用的隐藏层数量。"
        )

    # 选择最后 'num_layers_of_embedded' 层, 堆叠并平均
    features_avg = torch.stack(encoded_layers[-num_layers_of_embedded:], dim=1).mean(dim=1)

    # 复制您描述的除法缩放操作
    if num_layers_of_embedded > 0:
        features_scaled = features_avg / num_layers_of_embedded
    else:
        features_scaled = features_avg # 或者作为错误处理

    # features_scaled 形状是 (1, seq_len_padded, hidden_dim)

    # 为了进行 token 级别的选择，我们先移除 batch 维度 (因为 batch_size=1)
    squeezed_features_scaled = features_scaled.squeeze(0) # (seq_len_padded, hidden_dim)
    squeezed_input_ids = input_ids.squeeze(0)             # (seq_len_padded)
    squeezed_attention_mask_bert = attention_mask_bert.squeeze(0).bool() # (seq_len_padded)

    # 步骤 1: 根据 attention_mask_bert 移除填充 token
    # 只选择那些实际的（非填充）token 对应的特征和 ID
    actual_token_features = squeezed_features_scaled[squeezed_attention_mask_bert]
    actual_token_ids = squeezed_input_ids[squeezed_attention_mask_bert]
    # actual_token_features 形状: (actual_seq_len, hidden_dim)
    # actual_token_ids 形状: (actual_seq_len)

    # 步骤 2: 根据 actual_token_ids 移除点号 token
    dot_token_id = tokenizer.convert_tokens_to_ids('.')
    # 创建一个掩码，标记出在 actual_token_ids 中哪些不是点号
    dot_mask_for_actual_tokens = (actual_token_ids != dot_token_id)
    print(f"dot_mask_for_actual_tokens is",dot_mask_for_actual_tokens)
    # import sys
    # sys.exit()
    # 选择非点号 token 的特征
    features_no_dots = actual_token_features[dot_mask_for_actual_tokens]
    # features_no_dots 形状: (actual_seq_len_without_dots, hidden_dim)

    # 将特征重新构造成 (1, new_seq_len, hidden_dim)
    current_emb = features_no_dots.unsqueeze(0)

    # 步骤 3: 如果需要，移除 [CLS] 和 [SEP] token
    if remove_cls_sep:
        # 这个切片操作会移除当前序列的第一个和最后一个 token。
        # 它能正确处理序列长度为0, 1, 或 2 的情况，结果序列长度为0。
        current_emb = current_emb[:, 1:-1, :]
        
    return current_emb


def save_embedding(emb: torch.Tensor, save_dir: str, filename: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(emb, path)
    print(f"已保存嵌入至 {path}")

def load_embedding(save_dir: str, filename: str) -> torch.Tensor:
    path = os.path.join(save_dir, filename)
    return torch.load(path)

if __name__ == "__main__":
    #NOTE:这部分因为没法计算掩码之类的最终无法使用
    lang_model_path = '/home/wuke_2024/ov202503/text_encoder/bert-base-uncased' 
    background_text = "building.road.sidewalk.wall.fence.dirt.stone.sand.river.hill.tree.grass.bush.sky.cloud.rain.shadow.night.background.texture"
    num_bert_layers_to_avg = 1 
    used_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"使用语言模型: {lang_model_path}")
    print(f"平均最后 {num_bert_layers_to_avg} 个 BERT 层。")
    print(f"输入背景文本: \"{background_text}\"")
    print(f"使用设备: {used_device}")

    try:
        emb = get_bert_input_embedding(
            text=background_text,
            remove_cls_sep=True, 
            device=used_device,
            lang_model_name=lang_model_path,
            num_layers_of_embedded=num_bert_layers_to_avg
        )
        print("\n输出嵌入形状:", emb.shape)
        # 例如，如果输入是 "building.sky"，分词后可能是 "[CLS] building . sky [SEP]" (5 tokens)
        # 1. 移除填充 (假设无填充，还是5 tokens)
        # 2. 移除点号 "." (序列变为 "[CLS] building sky [SEP]", 4 tokens)
        # 3. 如果 remove_cls_sep=True, 移除 CLS 和 SEP (序列变为 "building sky", 2 tokens)
        # 最终形状会是 (1, 2, hidden_dim)

        emb_save_dir = "/home/wuke_2024/ov202503/mmdetection/ov_text_feature" 
        filename = "background_feature.pth"
        save_embedding(emb, emb_save_dir, filename)

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()