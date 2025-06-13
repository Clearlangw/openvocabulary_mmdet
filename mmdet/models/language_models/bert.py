# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Sequence,Optional,List

import torch
from mmengine.model import BaseModel
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
try:
    from transformers import AutoTokenizer, BertConfig
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

from mmdet.registry import MODELS
import os

class CoOpModule(nn.Module):
    def __init__(self,
        prompt_length: int=16,
        prompt_channel: int=768,
        coop_init: Optional[str] = None,
        coop_csc:bool=False,
        csc_cls_num:int=10,
        ) -> None:
        #这里是超极简实现，不具备CSC以及固定词init功能
        super().__init__()

        self.prompt_length = prompt_length
        self.prompt_channel = prompt_channel
        self.coop_init = coop_init
        self.coop_csc = coop_csc
        self.csc_cls_num = csc_cls_num
        if not self.coop_init and not self.coop_csc:
            self.coop_prompt = nn.Parameter(torch.zeros(1, self.prompt_length, self.prompt_channel))
            nn.init.normal_(self.coop_prompt, std=0.02)
        elif self.coop_init:
            #text = "An aerial photograph of a busy urban or park area, one of the objects is" #torch.Size([1, 16, 768])
            text = self.coop_init
            save_dir = "/home/wuke_2024/ov202503/mmdetection/ov_emb"
            filename = f"{text.replace(' ', '_')}.pth"
            coop_path = os.path.join(save_dir, filename)
            if os.path.exists(coop_path):
                self.coop_prompt = torch.load(coop_path)
                self.coop_prompt = nn.Parameter(self.coop_prompt)
                self.prompt_length = self.coop_prompt.shape[1]
            else:
                from ov_toolkits.coop_init import get_bert_input_embedding
                text = self.coop_init
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.coop_prompt = get_bert_input_embedding(text=text,remove_cls_seq=True,device=device)
                self.coop_prompt = nn.Parameter(self.coop_prompt)
                self.prompt_length = self.coop_prompt.shape[1]
        elif self.coop_csc:
            text="XX XX XX XX"
            save_dir = "/home/wuke_2024/ov202503/mmdetection/ov_emb"
            filename = f"{text.replace(' ', '_')}.pth"
            csc_coop_path = os.path.join(save_dir, filename)
            if os.path.exists(csc_coop_path):
                self.csc_coop_prompt = torch.load(csc_coop_path)
                self.coop_prompt_list = nn.Parameter(
                    self.csc_coop_prompt.unsqueeze(0).repeat(self.csc_cls_num, 1, 1, 1)
                )
                self.prompt_length = self.coop_prompt_list.shape[2]
            else:
                from ov_toolkits.coop_init import get_bert_input_embedding
                text = "XX XX XX XX"
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.csc_coop_prompt = get_bert_input_embedding(text=text,remove_cls_seq=True,device=device)
                self.coop_prompt_list = nn.Parameter(
                    self.csc_coop_prompt.unsqueeze(0).repeat(self.csc_cls_num, 1, 1, 1)
                )
                self.prompt_length = self.coop_prompt_list.shape[2]
    
    def forward(self, x):
        return x

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

def generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens_list,coop=None):
    """Generate attention mask between each pair of special tokens.

    Only token pairs in between two special tokens are attended to
    and thus the attention mask for these pairs is positive.

    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.

    Returns:
        Tuple(Tensor, Tensor):
        - attention_mask is the attention mask between each tokens.
          Only token pairs in between two special tokens are positive.
          Shape: [bs, num_token, num_token].
        - position_ids is the position id of tokens within each valid sentence.
          The id starts from 0 whenenver a special token is encountered.
          Shape: [bs, num_token]
    """
    input_ids = tokenized['input_ids']
    bs, num_token = input_ids.shape
    coop_prompt_length = coop.prompt_length if coop and not hasattr(coop,'csc_coop_prompt') else 0
    # print(input_ids)
    #TODO：第三部分：在 input_ids 的开头插入 coop.prompt_length 个值为 22953 的标记。即‘bro’
    if coop and not hasattr(coop,'csc_coop_prompt'):
        num_token += coop.prompt_length
        tokenized['input_ids'] = torch.cat((input_ids[:,0:1], torch.tensor([[22953 for _ in range(coop.prompt_length)] for _ in range(input_ids.shape[0])]).to(input_ids.device), input_ids[:, 1:]),dim=1)
        tokenized['token_type_ids'] = torch.cat((tokenized['token_type_ids'][:,0:1], torch.tensor([[0 for _ in range(coop.prompt_length)] for _ in range(tokenized['token_type_ids'].shape[0])]).to(input_ids.device), tokenized['token_type_ids'][:, 1:]),dim=1)
        input_ids = tokenized['input_ids']
    # special_tokens_mask:
    # bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token),
                                      device=input_ids.device).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)
    # print(idxs)
    # import sys
    # sys.exit()
    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token,
                  device=input_ids.device).bool().unsqueeze(0).repeat(
            bs, 1, 1))
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1,
            previous_col + 1:col + 1] = True
            # position_ids[row, previous_col + 1:col + 1] = torch.arange(
            #     0, col - previous_col, device=input_ids.device)
            #TODO：第四部分：在生成位置编码时，会跳过 CoOp 提示部分，确保位置编码的连续性。
            if previous_col==0:
                # print(col-previous_col-coop_prompt_length)
                # import sys
                # sys.exit()
                position_ids[row, previous_col + 1 + coop_prompt_length : col + 1] = torch.arange(
                    0, col - previous_col - coop_prompt_length, device=input_ids.device
                ) # skip the prompts
            else:
                position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                    0, col - previous_col, device=input_ids.device
                )
        previous_col = col

    #TODO：第六部分：注意力掩码中与 CoOp 提示部分对应的行和列都设置为 True ，表示这些位置之间可以相互关注。
    if coop and not hasattr(coop,'csc_coop_prompt'):
        attention_mask[:, 1:coop.prompt_length+1,:]=True
        attention_mask[:, :, 1:coop.prompt_length+1]=True

    return attention_mask, position_ids.to(torch.long)

@MODELS.register_module()
class BertModel(BaseModel):
    """BERT model for language embedding only encoder.

    Args:
        name (str, optional): name of the pretrained BERT model from
            HuggingFace. Defaults to bert-base-uncased.
        max_tokens (int, optional): maximum number of tokens to be
            used for BERT. Defaults to 256.
        pad_to_max (bool, optional): whether to pad the tokens to max_tokens.
             Defaults to True.
        use_sub_sentence_represent (bool, optional): whether to use sub
            sentence represent introduced in `Grounding DINO
            <https://arxiv.org/abs/2303.05499>`. Defaults to False.
        special_tokens_list (list, optional): special tokens used to split
            subsentence. It cannot be None when `use_sub_sentence_represent`
            is True. Defaults to None.
        add_pooling_layer (bool, optional): whether to adding pooling
            layer in bert encoder. Defaults to False.
        num_layers_of_embedded (int, optional): number of layers of
            the embedded model. Defaults to 1.
        use_checkpoint (bool, optional): whether to use gradient checkpointing.
             Defaults to False.
        requires_grad (bool,optional): whether to not frozen gradient.
                Defaults to False. #TODO:实现对bert的冻结
    """

    def __init__(self,
                 name: str = 'bert-base-uncased',
                 max_tokens: int = 256,
                 pad_to_max: bool = True,
                 use_sub_sentence_represent: bool = False,
                 special_tokens_list: list = None,
                 add_pooling_layer: bool = False,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False,
                 requires_grad: bool = False,
                 use_shine: bool = False,
                 shine_prompt_path: Optional[str] = None,
                 shine_prompt_cls_list: Optional[List[str]] = None,
                 shine_inplace_token_num: Optional[int] = None,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max

        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.language_backbone = nn.Sequential(
            OrderedDict([('body',
                          BertEncoder(
                              name,
                              add_pooling_layer=add_pooling_layer,
                              num_layers_of_embedded=num_layers_of_embedded,
                              use_checkpoint=use_checkpoint,requires_grad=requires_grad))]))

        # token_id = 22953
        # token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        # print(f"Token ID {token_id} 对应的内容是: '{token}'")
        # import sys
        # sys.exit()
        self.use_sub_sentence_represent = use_sub_sentence_represent
        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None, \
                'special_tokens should not be None \
                    if use_sub_sentence_represent is True'

            self.special_tokens = self.tokenizer.convert_tokens_to_ids(
                special_tokens_list)
        self.coop=None
        self.use_shine = use_shine
        self.shine_prompt_path = shine_prompt_path
        self.shine_prompt_cls_list = shine_prompt_cls_list
        self.shine_inplace_token_num = shine_inplace_token_num

    def add_coop_prompt(self,coop_prompt_length,coop_prompt_channel,coop_init,coop_csc,csc_cls_num=10):
        self.coop = CoOpModule(coop_prompt_length,coop_prompt_channel,coop_init,coop_csc,csc_cls_num)
        
    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        """Forward function."""
        device = next(self.language_backbone.parameters()).device
        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            max_length=self.max_tokens,
            padding='max_length' if self.pad_to_max else 'longest',
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True).to(device)
        input_ids = tokenized.input_ids

        #第1部分：实现变长的coop_prompt
        input_shape = input_ids.size()
        seq_length = input_shape[-1]

        if self.coop:
            # print("im here")
            # import sys
            # sys.exit()
            self.language_backbone.body.set_coop(self.coop)
            if not hasattr(self.coop, 'csc_coop_prompt'):
                seq_length = seq_length + self.coop.prompt_length
            

        # print(self.use_sub_sentence_represent) #True
        # import sys
        # sys.exit()
        #第二步，修改attention_masl,position_ids来修改tokenized，这里得谨慎一些。。。
        # 因为没摸清楚沟槽的tokenized调用
        # import pdb
        # pdb.set_trace()
        if self.use_sub_sentence_represent:
            attention_mask, position_ids = \
                generate_masks_with_special_tokens_and_transfer_map(
                    tokenized, self.special_tokens,self.coop)
            token_type_ids = tokenized['token_type_ids']

        else:
            attention_mask = tokenized.attention_mask
            position_ids = None
            token_type_ids = None

        tokenizer_input = {
            #'input_ids': input_ids, #这里被修改为使用tokenized的input_ids
            'input_ids': tokenized['input_ids'],
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }
        # print("___________________________________________________")
        # word_ids = tokenized.word_ids()
        # # 打印结果
        # tokens = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])
        # for i, (token, word_id) in enumerate(zip(tokens, word_ids)):
        #     print(f"Token {i}: {token}, belongs to word index: {word_id}")
        # import sys
        # sys.exit()
        # dot_token_id = self.tokenizer.convert_tokens_to_ids('.')
        # # 创建一个掩码，标记出在 actual_token_ids 中哪些不是点号
        # dot_mask_for_actual_tokens = (tokenized['input_ids'] != dot_token_id)
        # print(f"dot_mask_for_actual_tokens is",dot_mask_for_actual_tokens)
        # import sys
        # sys.exit()
        # print(f"tokenizer_input['input_ids'].shape is {tokenizer_input['input_ids'].shape} ")
        # print(f"tokenized['attention_mask'].shape is {tokenized['attention_mask'].shape} ") #[2,27]
        # print(f"tokenizer_input['attention_mask'].shape is {tokenizer_input['attention_mask'].shape} ")
        # print(f"tokenizer_input['position_ids'].shape is {tokenizer_input['position_ids'].shape} ")
        # print(f"tokenizer_input['token_type_ids'].shape is {tokenizer_input['token_type_ids'].shape} ")
        # print(f"tokenizer_input['token_type_ids'][0] is {tokenizer_input['token_type_ids'][0]} ")
        # import sys
        # sys.exit()
        #TODO：这里是bert之前，一切都可以挽回
        language_dict_features = self.language_backbone(tokenizer_input)
        if self.use_sub_sentence_represent: #GDINO走这条路，还原其实应该放在外边
            if self.coop and not hasattr(self.coop,'csc_coop_prompt'):
                language_dict_features['embedded'] = torch.cat((language_dict_features['embedded'][:,:1], language_dict_features['embedded'][:,1+self.coop.prompt_length:]), dim=1)
                position_ids = torch.cat((position_ids[:,:1], position_ids[:,1+self.coop.prompt_length:]), dim=1)        
                token_type_ids = torch.cat((tokenizer_input['token_type_ids'][:,:1], tokenizer_input['token_type_ids'][:,1+self.coop.prompt_length:]), dim=1)      
                #实际上bert的tokenizer的 attention_mask 是指示哪些 token 是有效的
                language_dict_features['masks'] = torch.cat((language_dict_features['masks'][:,:1], language_dict_features['masks'][:,1+self.coop.prompt_length:]),dim=1)
                language_dict_features['masks'] = torch.cat((language_dict_features['masks'][:,:,:1], language_dict_features['masks'][:,:,1+self.coop.prompt_length:]),dim=2)
            if self.use_shine:
                # print(f"language_dict_features['embedded'].shape is {language_dict_features['embedded'].shape}")
                #TODO:写个cache_path不用手动替换
                inplace_device = language_dict_features['embedded'].device
                current_pos = 1
                for class_name in self.shine_prompt_cls_list:
                    feature_filename = f"{class_name}_feature.pth"
                    feature_path = os.path.join(self.shine_prompt_path, feature_filename)
                    if not os.path.exists(feature_path):
                        print(f"  -> 警告: 未找到特征文件 {feature_path}，跳过该类别。")
                        # 更新位置以便下一个类别能被正确放置
                        current_pos += self.shine_inplace_token_num + 1
                        continue
                    try:
                        # 加载预计算的特征
                        loaded_feature = torch.load(feature_path, map_location=inplace_device)
                        # 验证加载的特征形状是否正确
                        expected_shape = (1, self.shine_inplace_token_num, language_dict_features['embedded'].shape[-1])
                        if loaded_feature.shape != expected_shape:
                            print(f"  -> 警告: 特征 {feature_filename} 形状不匹配。")
                            print(f"     期望形状: {expected_shape}, 实际形状: {loaded_feature.shape}")
                            current_pos += self.shine_inplace_token_num + 1
                            continue
                        # 定义要替换的目标切片
                        start_index = current_pos
                        end_index = current_pos + self.shine_inplace_token_num
                        # 执行替换操作
                        # PyTorch的广播机制会自动处理 batch size (bs)
                        language_dict_features['embedded'][:, start_index:end_index, :] = loaded_feature
                        # 为下一个类别更新起始位置
                        current_pos += self.shine_inplace_token_num + 1
                        # print(current_pos)
                        # print(f"替换完成")
                    except Exception as e:
                        print(f"  -> 处理文件 {feature_path} 时发生错误: {e}")
                        # 同样需要更新位置
                        current_pos += self.shine_inplace_token_num + 1
            language_dict_features['position_ids'] = position_ids
            language_dict_features[
                'text_token_mask'] = tokenized.attention_mask.bool()
            #NOTE:这里可以用来保存结果，因为保证了参数和模型原来的参数完全一致方便更改bert的参数
            # language_dict_features['dot_mask'] =  torch.tensor([ True,  True, False,  True, False,  True, False,  True, False,  True,
            #     False,  True, False,  True, False,  True, False,  True, False,  True,
            #     False,  True, False,  True, False,  True, False,  True, False,  True,
            #     False,  True, False,  True, False,  True, False,  True, False,  True,
            #     False,  True], device=language_dict_features['embedded'].device)
            # # print(language_dict_features.keys())
            # # import sys
            # # sys.exit()
            # torch.save(language_dict_features,'/home/wuke_2024/ov202503/mmdetection/ov_text_feature/back_dict.pth')
            # torch.save(language_dict_features['embedded'],'/home/wuke_2024/ov202503/mmdetection/ov_text_feature/visdrone_concept.pth')
            # import sys
            # sys.exit()
        return language_dict_features


class BertEncoder(nn.Module):
    """BERT encoder for language embedding.

    Args:
        name (str): name of the pretrained BERT model from HuggingFace.
                Defaults to bert-base-uncased.
        add_pooling_layer (bool): whether to add a pooling layer.
        num_layers_of_embedded (int): number of layers of the embedded model.
                Defaults to 1.
        use_checkpoint (bool): whether to use gradient checkpointing.
                Defaults to False.
        requires_grad (bool): whether to not frozen gradient.
                Defaults to False. #TODO:实现对bert的冻结
    """

    def __init__(self,
                 name: str,
                 add_pooling_layer: bool = False,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False,
                 requires_grad: bool = False):
        super().__init__()
        if BertConfig is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')
        config = BertConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint
        # only encoder
        self.model = HFBertModel.from_pretrained(
            name, add_pooling_layer=add_pooling_layer, config=config)
        # print(self.model._attr_)
        # import sys
        # sys.exit()
        for param in self.model.parameters():
            param.requires_grad = requires_grad

        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded
        self.coop = None

    def set_coop(self,coop):
        self.coop = coop

    def forward(self, x) -> dict:
        mask = x['attention_mask']
        # print(mask.shape)
        # import sys
        # sys.exit()
        ##
        # print(embedding_output.shape)
        # import sys
        # sys.exit()
        if self.coop:
            input_ids=x['input_ids']
            attention_mask=mask
            token_type_ids=x['token_type_ids']
            position_ids=x['position_ids']
            output_hidden_states=True
            head_mask = None
            inputs_embeds = None
            encoder_hidden_states = None
            encoder_attention_mask = None
            past_key_values = None
            use_cache = None
            output_attentions = None
            return_dict = None

            output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

            if self.model.config.is_decoder:
                use_cache = use_cache if use_cache is not None else self.model.config.use_cache
            else:
                use_cache = False

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                self.model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            batch_size, seq_length = input_shape
            device = input_ids.device if input_ids is not None else inputs_embeds.device

            # past_key_values_length
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            if token_type_ids is None:
                if hasattr(self.model.embeddings, "token_type_ids"):
                    buffered_token_type_ids = self.model.embeddings.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            embedding_output = self.model.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
            #TODO:这里做了改动
            if self.coop and not hasattr(self.coop, 'csc_coop_prompt'):
                coop_emb = self.coop.coop_prompt.to(embedding_output.device)
                coop_emb = coop_emb.repeat(embedding_output.size()[0], 1, 1) #复制维度
                embedding_output[:, 1:1+self.coop.prompt_length, :] = coop_emb
            if self.coop and hasattr(self.coop, 'csc_coop_prompt'):
                bs, seq_len, emb_dim = embedding_output.shape
                prompt_length = self.coop.prompt_length
                # 假设self.coop_prompt_list shape为 (bs, 1, prompt_length, emb_dim)
                # 如果是(num_cls, 1, prompt_length, emb_dim)，你可以用self.coop_prompt_list[i, 0]
                for i in range(bs):
                    pos_ids = position_ids[i].tolist()  # 转为list方便处理
                    segments = find_prompt_segments(pos_ids, prompt_length)
                    for seg_idx, (start, end) in enumerate(segments):
                        # 用第i个prompt参数组替换
                        # self.coop_prompt_list[i, 0, :, :] shape: [prompt_length, emb_dim]
                        embedding_output[i, start:end, :] = self.coop_prompt_list[i, 0, :, :]
            #考虑一下pos的emb
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

            use_sdpa_attention_masks = (
                self.model.attn_implementation == "sdpa"
                and self.model.position_embedding_type == "absolute"
                and head_mask is None
                and not output_attentions
            )

            # Expand the attention mask
            if use_sdpa_attention_masks and attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                if self.model.config.is_decoder:
                    extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                        attention_mask,
                        input_shape,
                        embedding_output,
                        past_key_values_length,
                    )
                else:
                    extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                        attention_mask, embedding_output.dtype, tgt_len=seq_length
                    )
            else:
                # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
                # ourselves in which case we just need to make it broadcastable to all heads.
                extended_attention_mask = self.model.get_extended_attention_mask(attention_mask, input_shape)

            # If a 2D or 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.model.config.is_decoder and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

                if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                    # Expand the attention mask for SDPA.
                    # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                    encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                        encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                    )
                else:
                    encoder_extended_attention_mask = self.model.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = None

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
            # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            head_mask = self.model.get_head_mask(head_mask, self.model.config.num_hidden_layers)

            encoder_outputs = self.model.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None

            if not return_dict:
                outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
            else:
                outputs=BaseModelOutputWithPoolingAndCrossAttentions(
                    last_hidden_state=sequence_output,
                    pooler_output=pooled_output,
                    past_key_values=encoder_outputs.past_key_values,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                    cross_attentions=encoder_outputs.cross_attentions,
                )
        else:
            #  原来的一体化实现
            outputs = self.model(
                input_ids=x['input_ids'],
                attention_mask=mask,
                position_ids=x['position_ids'],
                token_type_ids=x['token_type_ids'],
                output_hidden_states=True,
            )

        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-self.num_layers_of_embedded:],
                               1).mean(1)
        # language embedding has shape [len(phrase), seq_len, language_dim]
        features = features / self.num_layers_of_embedded
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features
        # print("embedded.shape is",embedded.shape) #[bs=2,token_len=27,language_dim=768]
        # print("embeded is",embedded)
        # mask = torch.tensor([ True,  True, False,  True, False,  True, False,  True, False,  True,
        # False,  True, False,  True, False,  True, False,  True, False,  True,
        # False,  True, False,  True, False,  True, False,  True, False,  True,
        # False,  True, False,  True, False,  True, False,  True, False,  True,
        # False,  True], device=embedded.device)
        # # 确保mask的维度与tokens维度匹配
        # assert mask.size(0) == embedded.size(1), f"Mask length {mask.size(0)} doesn't match token length {embedded.size(1)}"
        # # 使用布尔索引选择True位置的特征
        # embedded = embedded[:, mask, :]
        # # 移除padding
        # embedded = embedded[:,1:-1,:]
        # path = "/home/wuke_2024/ov202503/mmdetection/ov_text_feature/dino_bert_background_feature.pth"
        # torch.save(embedded, path)
        # print(f"embedded.shape is",embedded.shape)
        # import sys
        # sys.exit()
        results = {
            'embedded': embedded,
            'masks': mask,
            'hidden': encoded_layers[-1]
        }
        return results
