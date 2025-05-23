import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from typing import Optional

def get_bert_input_embedding(text: str, 
                            remove_cls_seq:bool = True,
                            # channel: int = 768,
                            device: str = 'cuda:8',
                            lang_model_name:str = '/home/wuke_2024/ov202503/text_encoder/bert-base-uncased',
                            ):
    tokenizer = BertTokenizer.from_pretrained(lang_model_name)
    model = BertModel.from_pretrained(lang_model_name).to(device)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']  # (1, seq_len)
    print(input_ids)
    # 只经过 embedding 层
    # with torch.no_grad():
    #     #emb = model.embeddings.word_embeddings(input_ids)
    #     emb = model.embeddings(input_ids)  # (1, seq_len, channel)
    dot_token_id = tokenizer.encode('.')[1]  # 获取点号的token id
    print(dot_token_id)
    mask = (input_ids != dot_token_id)  # 创建mask，True表示非点号位置
    
    with torch.no_grad():
        emb = model.embeddings(input_ids)
        # 使用mask过滤掉点号对应的embedding
        emb = emb[mask.unsqueeze(-1).expand_as(emb)].view(1, -1, emb.size(-1))
    if remove_cls_seq:
        emb = emb[:,1:-1,:]
    return emb  # shape: (1, seq_len, channel)


def save_embedding(emb: torch.Tensor, save_dir: str, filename: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(emb, path)

def load_embedding(save_dir: str, filename: str) -> torch.Tensor:
    path = os.path.join(save_dir, filename)
    return torch.load(path)

def find_prompt_segments(pos_ids, prompt_length):
    """
    pos_ids: 1D tensor/list, e.g. [0,0,0,1,2,3,4,0,1,2,3,4,5]
    prompt_length: int
    return: list of (start, end) indices for each complete [0,1,...,prompt_length-1] segment
    """
    segments = []
    i = 0
    while i <= len(pos_ids) - prompt_length:
        # 检查当前位置是否为0，且后面连续prompt_length个数正好是[0,1,...,prompt_length-1]
        if all(pos_ids[i+j] == j for j in range(prompt_length)):
            segments.append((i, i+prompt_length))  # [start, end)
            i += prompt_length  # 跳过这个段
        else:
            i += 1
    return segments

if __name__ == "__main__":
    # text = "An aerial photograph of a busy urban or park area, one of the objects is" #torch.Size([1, 16, 768])
    # text="XX XX XX XX XX"
    text = "building.road.sidewalk.wall.fence.dirt.stone.sand.river.hill.tree.grass.bush.sky.cloud.rain.shadow.night.background.texture"
    # text = "an aerial image of" #torch.Size([1, 4, 768])
    # text = "an aerial view image of" #torch.Size([1, 5, 768])
    emb_save_dir = "/home/wuke_2024/ov202503/mmdetection/ov_emb"
    emb = get_bert_input_embedding(text)
    filename = f"{text.replace(' ', '_')}.pth"
    filename = "background_wo_dot.pth"
    #emb_path = os.path.join(emb_save_dir, filename)
    save_embedding(emb,emb_save_dir,filename)

    print(emb.shape)
    print(emb)


    # # 示例1：随机初始化
    # print("=== Example 1: Random Init ===")
    # model1 = CoOpModule(prompt_length=10, prompt_channel=768, coop_init=None)
    # print(model1.forward().shape)  # torch.Size([1, 10, 768])
    # # 示例2：用指定文本初始化
    # print("\n=== Example 2: Text Init ===")
    # text = "a photo of a cat"
    # model2 = CoOpModule(prompt_channel=768, coop_init=text)
    # print(model2.forward().shape)  # torch.Size([1, token数, 768])