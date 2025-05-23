# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)


def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_

# class CoOpModule(nn.Module):
#     def __init__(self,
#         prompt_length: int=16,
#         prompt_channel: int=768) -> None:
#         #这里是超极简实现，不具备CSC以及固定词init功能
#         super().__init__()

#         self.prompt_length = prompt_length
#         self.prompt_channel = prompt_channel
#         self.coop_prompt = nn.Parameter(torch.zeros(1, self.prompt_length, self.prompt_channel))
#         nn.init.normal_(self.coop_prompt, std=0.02)
    
#     def forward(self, x):
#         return x


@MODELS.register_module()
class GroundingDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 use_coop=False,
                 use_fake_coop=False,
                 use_cocoop=False,
                 use_maple=False,
                 coop_init=None,
                 coop_csc=False,
                 background_supp=False, #背景抑制background_suppression，在predecoder那里修改
                 background_supp_mode = 1,
                 use_mona = False, #mona微调，使用mona对swintransformer进行微调
                 num_tokens=27,
                 *args,
                 use_autocast=False,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        assert sum([use_coop, use_cocoop, use_maple]) <= 1, "Only one of 'use_coop', 'use_cocoop', or 'use_maple' can be True at a time."
        #text端：
        self.use_coop=use_coop
        self.use_fake_coop = use_fake_coop
        self.use_cocoop=use_cocoop
        self.use_maple=use_maple
        self.coop_init=coop_init
        self.coop_csc =coop_csc #是否每一类均有不同的上下文参数，仅参考coop原论文
        #coop相关参数
        #TODO:实际to_enhance_text_prompts有前后缀的实现，可以merge
        self.coop_prompt_length=self.coop_n_ctx = 16 #可训练参数长度(左命名参考mrgdino，右名参考coop原论文)
        self.coop_prompt_channel=self.coop_ctx_dim  = 768
        self.num_tokens = num_tokens #用于告诉coop微调词的数量
        #vis端：swin微调参数
        self.use_mona = use_mona
        #VL交互：
        self.background_supp = background_supp
        self.background_supp_mode = background_supp_mode
        self.classnames = None #

        self._special_tokens = '. '
        self.use_autocast = use_autocast
        super().__init__(*args, **kwargs)
        

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        #初始化coop
        if self.use_coop:
            self.language_model.add_coop_prompt(self.coop_prompt_length,self.coop_prompt_channel,self.coop_init,self.coop_csc)
        if self.use_fake_coop:
            self.fake_coop = nn.Parameter(torch.zeros(1, self.num_tokens, self.coop_prompt_channel))
            nn.init.normal_(self.fake_coop, std=0.02)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            #print('caption_string is ',caption_string)   #pedestrian. people. bicycle. car. van. truck. tricycle. awning-tricycle. bus. motor.
            # import sys
            # sys.exit()
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt') #输出27个token
            # print("___________________________________________________")
            # word_ids = tokenized.word_ids()
            # # 打印结果
            # tokens = self.language_model.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])
            # for i, (token, word_id) in enumerate(zip(tokens, word_ids)):
            #     print(f"Token {i}: {token}, belongs to word index: {word_id}")

            # Token 0: [CLS], belongs to word index: None
            # Token 1: pedestrian, belongs to word index: 0
            # Token 2: ., belongs to word index: 1
            # Token 3: people, belongs to word index: 2
            # Token 4: ., belongs to word index: 3
            # Token 5: bicycle, belongs to word index: 4
            # Token 6: ., belongs to word index: 5
            # Token 7: car, belongs to word index: 6
            # Token 8: ., belongs to word index: 7
            # Token 9: van, belongs to word index: 8
            # Token 10: ., belongs to word index: 9
            # Token 11: truck, belongs to word index: 10
            # Token 12: ., belongs to word index: 11
            # Token 13: tri, belongs to word index: 12
            # Token 14: ##cycle, belongs to word index: 12
            # Token 15: ., belongs to word index: 13
            # Token 16: aw, belongs to word index: 14
            # Token 17: ##ning, belongs to word index: 14
            # Token 18: -, belongs to word index: 15
            # Token 19: tri, belongs to word index: 16
            # Token 20: ##cycle, belongs to word index: 16
            # Token 21: ., belongs to word index: 17
            # Token 22: bus, belongs to word index: 18
            # Token 23: ., belongs to word index: 19
            # Token 24: motor, belongs to word index: 20
            # Token 25: ., belongs to word index: 21
            # Token 26: [SEP], belongs to word index: None
            # import sys
            # sys.exit()

            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption) #名词提取
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        #这里是总的
        # The forward procedure of the transformer is defined as:
        # 'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        #  dict: The dictionary of encoder outputs, which includes the
        #  `memory` of the encoder output.

        # print("memory shape is",memory.shape) #([2, 16320, 256])
        # print("memory text shape is",memory_text.shape) #([2,27,256])
        # import sys
        # sys.exit()

        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # 用于top-k选择的原始得分 (每个候选框在所有文本token/类别上的最大得分)
        # original_scores 形状: (bs, num_all_proposals)
        num_all_proposals = output_proposals.shape[1]
        original_scores = enc_outputs_class.max(-1)[0]
        alpha = 0.5*1e-2              # 用于 'score_modulation',最初为0.5
        m_factor = 2               # 用于 'reranking',最初为2
        beta = 0.5*1e-2               # 用于 'reranking',最初为0.5  
        similarity_threshold = 150.0 # 用于 'gating' (需要根据点积的范围调整) 最开始为10.0
        penalty_value = 1e5       # 用于 'gating'
        # --- 背景抑制逻辑 ---
        background_text_feat = torch.load('/home/wuke_2024/ov202503/mmdetection/ov_emb/background_wo_dot.pth',map_location=output_memory.device)
        if background_text_feat.dim() == 3 and background_text_feat.shape[0] == 1:
            background_text_feat = background_text_feat.squeeze(0)
        # 扩展batch维度
        background_text_feat = background_text_feat.unsqueeze(0).expand(bs, -1, -1)
        background_text_feat=self.text_feat_map(background_text_feat)
        if self.background_supp and background_text_feat is not None:
            # 准备 background_text_feat
            # 预期形状: (bs, num_bg_tokens, c_feat) 或 (1, num_bg_tokens, c_feat)
            # 也可接受 (num_bg_tokens, c_feat), 将其视为 bs=1
            processed_bg_feat = background_text_feat
            if processed_bg_feat.ndim == 2: # (num_bg_tokens, c_feat)
                # 假设 bs=1 或全局背景token集合, 扩展为 (1, num_bg_tokens, c_feat)
                processed_bg_feat = processed_bg_feat.unsqueeze(0)
            
            if processed_bg_feat.ndim != 3:
                raise ValueError(
                    f"background_text_feat has unexpected ndim ({processed_bg_feat.ndim}) after potential unsqueeze. Expected 3 (e.g., bs/1, num_bg_tokens, c_feat)."
                )
            
            # num_bg_tokens = processed_bg_feat.shape[1]
            # 计算 proposal 特征与每个背景 token 特征的点积相似度
            # output_memory_proposals_feat: (bs, num_all_proposals, c_feat)
            # processed_bg_feat.transpose(-1, -2): (bs_bg, c_feat, num_bg_tokens) where bs_bg can be 1 or bs
            # proposal_bg_token_sim_matrix 形状: (bs, num_all_proposals, num_bg_tokens)
            # torch.matmul 会自动处理批次大小的广播 (如果 processed_bg_feat 的 bs_bg=1 且 output_memory_proposals_feat 的 bs > 1)
            proposal_bg_token_sim_matrix = torch.matmul(output_memory, 
                                                        processed_bg_feat.transpose(-1, -2))
            
            # 对每个 proposal，取其与所有背景 token 相似度中的最大值
            # background_similarity 形状: (bs, num_all_proposals)
            background_similarity = torch.max(proposal_bg_token_sim_matrix, dim=-1)[0]
            # # 验证相似度(zero shot下示例)
            # print(f'background_similarity is',background_similarity) #相似度 tensor([[151.4507, 151.4507, 151.4507,  ..., 180.0379, 173.8595, 160.4793]],
            # print(f'original_scores is',original_scores) #最初分数 tensor([[-5.0210, -5.0210, -5.0210,  ..., -5.0174, -5.0957, -5.2854]],
            # background_mask = background_similarity > similarity_threshold
            # print(f'background_mask is',background_mask)
            # # background_similarity min: 42.2682, max: 258.1486
            # # original_scores min: -7.0973, max: -0.1072
            # print(f'background_similarity min: {background_similarity.min().item():.4f}, max: {background_similarity.max().item():.4f}')
            # print(f'original_scores min: {original_scores.min().item():.4f}, max: {original_scores.max().item():.4f}')
            # print(f'background_mask True count: {background_mask.sum().item()}')
            # print(f'background_mask False count: {(~background_mask).sum().item()}')
            # import numpy as np
            # # 假设 background_similarity, original_scores 都是 [bs, num_proposals] 的 tensor
            # np.save('/home/wuke_2024/ov202503/mmdetection/zero_shot_background_similarity.npy', background_similarity.cpu().numpy())
            # np.save('/home/wuke_2024/ov202503/mmdetection/zero_shot_original_scores.npy', original_scores.cpu().numpy())
            # import sys
            # sys.exit()
            # # 验证相似度(full training下示例)
            # print(f'background_similarity is',background_similarity) #相似度 tensor([[146.5329, 146.5329, 146.5329,  ..., 160.8054, 160.1912, 137.2802]],
            # print(f'original_scores is',original_scores) #最初分数 tensor([[-4.5432, -4.5432, -4.5432,  ..., -5.1797, -5.1553, -4.9001]],
            # background_mask = background_similarity > similarity_threshold
            # print(f'background_mask is',background_mask)
            # # 统计信息
            # # background_similarity min: 44.5390, max: 275.4955
            # # original_scores min: -7.1635, max: 1.3547
            # print(f'background_similarity min: {background_similarity.min().item():.4f}, max: {background_similarity.max().item():.4f}')
            # print(f'original_scores min: {original_scores.min().item():.4f}, max: {original_scores.max().item():.4f}')
            # print(f'background_mask True count: {background_mask.sum().item()}')
            # print(f'background_mask False count: {(~background_mask).sum().item()}')
            # import numpy as np
            # # 假设 background_similarity, original_scores 都是 [bs, num_proposals] 的 tensor
            # np.save('/home/wuke_2024/ov202503/mmdetection/full_training_background_similarity.npy', background_similarity.cpu().numpy())
            # np.save('/home/wuke_2024/ov202503/mmdetection/full_training_original_scores.npy', original_scores.cpu().numpy())
            # import sys
            # sys.exit()
            if self.background_supp_mode == 'score_modulation':
                # 方案1: 用背景相似度调整原始得分
                modulated_scores = original_scores - alpha * background_similarity
                topk_indices = torch.topk(modulated_scores, k=self.num_queries, dim=1)[1]
                # print(f"使用score_modulation。原始最高分: {original_scores.max().item()}, 调整后最高分: {modulated_scores.max().item()}")

            elif self.background_supp_mode == 'reranking':
                # 方案2: 选择top-M, 然后使用背景相似度进行重排序
                num_queries_k = self.num_queries
                num_queries_m = min(num_queries_k * m_factor, num_all_proposals)

                top_m_original_scores, top_m_indices = torch.topk(
                    original_scores, k=num_queries_m, dim=1)
                
                top_m_output_memory_feat = torch.gather(
                    output_memory, 1,
                    top_m_indices.unsqueeze(-1).repeat(1, 1, c))

                # 计算这M个候选框与背景token的最大相似度
                # top_m_output_memory_feat: (bs, M, c_feat)
                # proposal_m_bg_token_sim_matrix 形状: (bs, M, num_bg_tokens)
                proposal_m_bg_token_sim_matrix = torch.matmul(top_m_output_memory_feat,
                                                              processed_bg_feat.transpose(-1, -2))
                background_similarity_m = torch.max(proposal_m_bg_token_sim_matrix, dim=-1)[0] # (bs, M)
                
                reranked_scores_m = top_m_original_scores - beta * background_similarity_m
                _, final_topk_indices_in_m = torch.topk(
                    reranked_scores_m, k=num_queries_k, dim=1)
                topk_indices = torch.gather(top_m_indices, 1, final_topk_indices_in_m)
                # print(f"使用reranking。M={num_queries_m}, K={num_queries_k}")

            elif self.background_supp_mode == 'gating':
                # 方案3: 惩罚背景相似度高于阈值的候选框
                background_mask = background_similarity > similarity_threshold
                penalized_scores = original_scores - background_mask.float() * penalty_value
                topk_indices = torch.topk(penalized_scores, k=self.num_queries, dim=1)[1]
                # print(f"使用gating。被掩码的大致数量: {background_mask.sum().item() / bs}")

            else: 
                topk_indices = torch.topk(original_scores, k=self.num_queries, dim=1)[1]
        else:
            # 没有背景抑制或未提供 background_text_feat
            topk_indices = torch.topk(original_scores, k=self.num_queries, dim=1)[1]
            # if self.background_supp and background_text_feat is None:
            #    print("背景抑制已启用, 但 background_text_feat 为 None。跳过。")
        # --- 背景抑制逻辑结束 ---

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.

        # topk_indices = torch.topk(
        #     enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()


        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]
        # print('tokens_positive' in batch_data_samples[0]) #False
        # print(text_prompts[0]) #('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
        # import sys
        # sys.exit()
        assert 'tokens_positive' not in batch_data_samples[0], \
    "Error: tokens_positive should not exist in batch_data_samples[0] during ov-detection"
        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                #TODO:实现有coop填充的token化，之后需要删除没必要的影响
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                #
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            # print('text_prompts are',text_prompts) #(bs,class)
            # import sys
            # sys.exit()
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                # print('tokenized is ',tokenized) #27个token，why tokenizer的正常行为，会有映射指向token和单词的关系
                # print('tokenized[input_ids] is ',tokenized['input_ids'].shape) #27
                #print('caption_string is ',caption_string) #pedestrian. people. bicycle. car. van. truck. tricycle. awning-tricycle. bus. motor.
                #print("tokens_positive is",tokens_positive) #tokens_positive is [[[0, 10]], [[12, 18]], [[20, 27]], [[29, 32]], [[34, 37]], [[39, 44]], [[46, 54]], [[56, 71]], [[73, 76]], [[78, 83]]]
                #tokens_positive:char哪些位置有效
                # import sys
                # sys.exit()
                new_text_prompts = [caption_string] * len(batch_inputs)
                # print("gt labels are ",gt_labels) #(bs,每个label)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    #positive_map_label_to_token, positive_map 
                    positive_maps.append(positive_map)
                # print('positive maps: ',positive_maps) #positive_map 
                # # print('positive maps[0]: ',positive_maps[0])
                # print('positive maps[0].shape: ',positive_maps[0].shape) #[gt,256]
                # print('positive maps[0][0]: ',positive_maps[0][0])
                # print('new_tokens_positive: ',new_tokens_positive)
                # import sys
                # sys.exit()
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)
            # print('new_text_prompts are: ',new_text_prompts)
            # new_text_prompts are:  ['pedestrian. people. bicycle. car. van. truck. tricycle. awning-tricycle. bus. motor. ', 
            # 'pedestrian. people. bicycle. car. van. truck. tricycle. awning-tricycle. bus. motor. ']
            # import sys
            # sys.exit()

        # if self.use_coop:
        #     #把token替换为这部分
        text_dict = self.language_model(new_text_prompts)

        # input_ids 负责表示输入文本的内容， position_ids 负责提供词元的位置信息，，每个token在每次词中间的位置
        # 而 mask 则负责处理填充问题，确保模型能够正确处理批量输入每个词只关心自己部分的token
        # for key in text_dict.keys():
        #     print(f"text_dict's {key} is {text_dict[key]}")
        #     print("*"*100)
        #     print(f"shape of text_dict's {key} is {text_dict[key].shape}")
        # import sys
        # sys.exit()

        # #TODO：Coop在GDINO第四处实现，文本还原为原本的编码文本特征
        # if self.coop:
        #      text_dict['embedded'] = torch.cat((text_dict['embedded'][:,:1], text_dict['embedded'][:,1+self.coop_prompt_length:]), dim=1)
        if self.use_fake_coop:
            fake_coop = self.fake_coop.repeat(text_dict['embedded'].size()[0], 1, 1)
            text_dict['embedded'] = fake_coop+text_dict['embedded']
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        

        # #TODO：Coop早GDINO第五处实现，还原att
        # if self.coop:
        #     position_ids = torch.cat((position_ids[:,:1], position_ids[:,17:]), dim=1)
            
        #     text_self_attention_masks = torch.cat((text_self_attention_masks[:,:1], text_self_attention_masks[:,17:]),dim=1)
            
        #     text_self_attention_masks = torch.cat((text_self_attention_masks[:,:,:1], text_self_attention_masks[:,:,17:]),dim=2)
        # print(self.text_feat_map)#linear[768,256]
        # print('text_dict[embedded] is ',text_dict['embedded'])#没有归一化的一个变量
        # print('text_dict[embedded] shape is ',text_dict['embedded'].shape) #[bs=2,token=27,dim=256]
        # import sys
        # sys.exit()
        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        #embv [2,27,256]
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)
        # print(isinstance(text_prompts[0], list)) #False
        # print(text_prompts[0]) #经典
        # import sys
        # sys.exit()
        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                # print(f"text_prompts_once is {text_prompts_once}")
                # import sys
                # sys.exit()
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.use_fake_coop:
                    fake_coop = self.fake_coop.repeat(text_dict['embedded'].size()[0], 1, 1)
                    text_dict['embedded'] = fake_coop+text_dict['embedded']
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            #还是走这里的
            # print(f"text_prompts_once is {text_prompts_once}")
            # import sys
            # sys.exit()
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.use_fake_coop:
                # print(f'self.use_fake_coop is {self.use_fake_coop}')
                # import sys
                # sys.exit()
                fake_coop = self.fake_coop.repeat(text_dict['embedded'].size()[0], 1, 1)
                text_dict['embedded'] = fake_coop+text_dict['embedded']
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples

# class CoopPromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         n_cls = len(classnames) #类别数量
#         n_ctx = cfg.TRAINER.COOP.N_CTX #提示长度
#         ctx_init = cfg.TRAINER.COOP.CTX_INIT #无，可以忽视
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         clip_imsize = clip_model.visual.input_resolution
#         cfg_imsize = cfg.INPUT.SIZE[0]
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

#         if ctx_init:#"xxxxxx"
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#             prompt = clip.tokenize(ctx_init)
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
#             #切片[1: 1 + n_ctx]选择了除起始token之外的n_ctx个token对应的嵌入向量
#             prompt_prefix = ctx_init

#         else:
#             # random initialization
#             if cfg.TRAINER.COOP.CSC:
#                 print("Initializing class-specific contexts")
#                 ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#             else:
#                 print("Initializing a generic context")
#                 ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)

#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")

#         self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]

#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens
#         self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

#     def forward(self):
#         ctx = self.ctx
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

#         prefix = self.token_prefix
#         suffix = self.token_suffix

#         if self.class_token_position == "end":
#             prompts = torch.cat(
#                 [
#                     prefix,  # (n_cls, 1, dim)
#                     ctx,     # (n_cls, n_ctx, dim)
#                     suffix,  # (n_cls, *, dim)
#                 ],
#                 dim=1,
#             )

#         elif self.class_token_position == "middle":
#             half_n_ctx = self.n_ctx // 2
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i : i + 1, :, :]
#                 class_i = suffix[i : i + 1, :name_len, :]
#                 suffix_i = suffix[i : i + 1, name_len:, :]
#                 ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
#                 ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,     # (1, 1, dim)
#                         ctx_i_half1,  # (1, n_ctx//2, dim)
#                         class_i,      # (1, name_len, dim)
#                         ctx_i_half2,  # (1, n_ctx//2, dim)
#                         suffix_i,     # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)

#         elif self.class_token_position == "front":
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i : i + 1, :, :]
#                 class_i = suffix[i : i + 1, :name_len, :]
#                 suffix_i = suffix[i : i + 1, name_len:, :]
#                 ctx_i = ctx[i : i + 1, :, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         class_i,   # (1, name_len, dim)
#                         ctx_i,     # (1, n_ctx, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)

#         else:
#             raise ValueError

#         return prompts


# class CocoopPromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = cfg.TRAINER.COCOOP.N_CTX
#         ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         vis_dim = clip_model.visual.output_dim
#         clip_imsize = clip_model.visual.input_resolution
#         cfg_imsize = cfg.INPUT.SIZE[0]
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

#         if ctx_init:
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#             prompt = clip.tokenize(ctx_init)
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
#             prompt_prefix = ctx_init
#         else:
#             # random initialization
#             ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)

#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")

#         self.ctx = nn.Parameter(ctx_vectors)

#         self.meta_net = nn.Sequential(OrderedDict([
#             ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
#             ("relu", nn.ReLU(inplace=True)),
#             ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
#         ]))

#         if cfg.TRAINER.COCOOP.PREC == "fp16":
#             self.meta_net.half()

#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]

#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens

#     def construct_prompts(self, ctx, prefix, suffix, label=None):
#         # dim0 is either batch_size (during training) or n_cls (during testing)
#         # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
#         # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
#         # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

#         if label is not None:
#             prefix = prefix[label]
#             suffix = suffix[label]

#         prompts = torch.cat(
#             [
#                 prefix,  # (dim0, 1, dim)
#                 ctx,  # (dim0, n_ctx, dim)
#                 suffix,  # (dim0, *, dim)
#             ],
#             dim=1,
#         )

#         return prompts

#     def forward(self, im_features):
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#         ctx = self.ctx  # (n_ctx, ctx_dim)
#         bias = self.meta_net(im_features)  # (batch, ctx_dim)
#         bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
#         ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
#         ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

#         # Use instance-conditioned context tokens for all classes
#         prompts = []
#         for ctx_shifted_i in ctx_shifted:
#             ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
#             pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
#             prompts.append(pts_i)
#         prompts = torch.stack(prompts)

#         return prompts


# class MultiModalPromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = cfg.TRAINER.MAPLE.N_CTX
#         ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         clip_imsize = clip_model.visual.input_resolution
#         cfg_imsize = cfg.INPUT.SIZE[0]
#         # Default is 1, which is compound shallow prompting
#         assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
#         self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

#         if ctx_init and (n_ctx) <= 4:
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = n_ctx
#             prompt = clip.tokenize(ctx_init)
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
#             prompt_prefix = ctx_init
#         else:
#             # random initialization
#             ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)
#         print('MaPLe design: Multi-modal Prompt Learning')
#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of MaPLe context words (tokens): {n_ctx}")
#         # These below, related to the shallow prompts
#         # Linear layer so that the tokens will project to 512 and will be initialized from 768
#         self.proj = nn.Linear(ctx_dim, 768)
#         self.proj.half()
#         self.ctx = nn.Parameter(ctx_vectors)
#         # These below parameters related to the shared prompts
#         # Define the compound prompts for the deeper layers

#         # Minimum can be 1, which defaults to shallow MaPLe
#         # compound prompts
#         self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
#                                                       for _ in range(self.compound_prompts_depth - 1)])
#         for single_para in self.compound_prompts_text:
#             nn.init.normal_(single_para, std=0.02)
#         # Also make corresponding projection layers, for each prompt
#         single_layer = nn.Linear(ctx_dim, 768)
#         self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]

#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens

#     def construct_prompts(self, ctx, prefix, suffix, label=None):
#         # dim0 is either batch_size (during training) or n_cls (during testing)
#         # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
#         # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
#         # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

#         if label is not None:
#             prefix = prefix[label]
#             suffix = suffix[label]

#         prompts = torch.cat(
#             [
#                 prefix,  # (dim0, 1, dim)
#                 ctx,  # (dim0, n_ctx, dim)
#                 suffix,  # (dim0, *, dim)
#             ],
#             dim=1,
#         )

#         return prompts

#     def forward(self):
#         ctx = self.ctx

#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

#         prefix = self.token_prefix
#         suffix = self.token_suffix
#         prompts = self.construct_prompts(ctx, prefix, suffix)

#         # Before returning, need to transform
#         # prompts to 768 for the visual side
#         visual_deep_prompts = []
#         for index, layer in enumerate(self.compound_prompt_projections):
#             visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
#         # Now the other way around
#         # We will project the textual prompts from 512 to 768
#         return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required

##implement from MRGDINO

# from timm.models.layers import trunc_normal_

# lan_scale = 0.1
# vis_scale = 0.1



# class RepZeroLinear(nn.Linear):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
#         super().__init__(in_features, out_features, bias, device, dtype)
#         self.scaling = nn.parameter.Parameter(torch.ones(1) * lan_scale)
#         nn.init.constant_(self.weight, val=zero_value)
#         self.freeze_linear = nn.Linear(in_features, out_features, bias, device, dtype)
#         nn.init.constant_(self.freeze_linear.weight, val=0.0)
#         if self.bias is not None:
#             nn.init.constant_(self.freeze_linear.bias, val=0.0) 
        
#         # self.zero_inter_loss = torch.nn.L1Loss(reduction='mean')
#         # self.zero_inter_loss = torch.nn.MSELoss(reduction='mean')
#         self.zero_inter_loss = torch.nn.SmoothL1Loss(reduction='mean')

#     def forward(self, input: Tensor) -> Tensor:
#         if self.training:
#             branch_output = self.scaling * super().forward(input)
#             output = branch_output + self.freeze_linear(input)
#             return output, \
#                 self.zero_inter_loss(branch_output, torch.zeros_like(branch_output)) + \
#                     self.zero_inter_loss(output, torch.zeros_like(output))
#         else:
#             return self.freeze_linear(input), torch.zeros(1).to(input)

#     def __rep__(self):
#         self.freeze_linear.weight.data = self.weight.data  * self.scaling + self.freeze_linear.weight.data
#         self.freeze_linear.bias.data = self.bias.data  * self.scaling + self.freeze_linear.bias.data
#         self.scaling = nn.parameter.Parameter(torch.ones(1).to(self.weight.data) * lan_scale)
#         nn.init.constant_(self.weight, val=zero_value)
#         if self.bias is not None:
#             nn.init.constant_(self.bias, val=zero_value)
            
            


# def shift_columns(tensor, m, k):
#     """
#     将前 m 列整体向右移动 k 列，并保持剩下的列为 -inf
#     参数:
#     - tensor: 需要操作的二维 tensor
#     - m: 前 m 列是非-inf 的部分
#     - k: 向右移动的列数
#     """
#     # 获取 tensor 的行数和列数
#     num_rows, num_cols = tensor.shape

#     # 创建新的 tensor，初始化为 -inf
#     new_tensor = torch.full((num_rows, num_cols), float('-inf'))

#     # 计算移动后有效列的起始索引和终止索引
#     start_idx = k
#     end_idx = min(k + m, num_cols)  # 确保不会超出边界

#     # 将原 tensor 的前 m 列复制到新的 tensor 的 [k:k+m] 位置
#     new_tensor[:, start_idx:end_idx] = tensor[:, :end_idx - k]

#     return new_tensor


# def find_inf_boundary(t):
#     # 创建一个布尔掩码，检查每个元素是否为 -inf
#     is_inf = t == float('-inf')
    
#     # 对每一列求和，看看是否整列都为 -inf
#     # 使用 all(dim=0) 来检查每一列是否都是 True (-inf)
#     inf_cols = is_inf.all(dim=0)
    
#     # 返回第一个全为 -inf 的列的索引
#     inf_boundary_index = torch.nonzero(inf_cols).min().item()
    
#     return inf_boundary_index


# class CoOpModule(nn.Module):
#     def __init__(self, prompt_length, prompt_channel, use_prompt=False, prompt=None) -> None:
#         super().__init__()
#         self.prompt_length = prompt_length
#         self.prompt_channel = prompt_channel
#         if use_prompt:
#             self.coop_prompt = prompt
#         else:
#             self.coop_prompt = nn.Parameter(torch.zeros(1, self.prompt_length, self.prompt_channel))
#             trunc_normal_(self.coop_prompt, std=0.02)
    
#     def forward(self, x):
#         return x



# class Prompt(nn.Module):
#     def __init__(self, length=4, embed_dim=768, embed_dim_key=768, embedding_key='mean', prompt_init='uniform', prompt_pool=True, 
#                  prompt_key=True, pool_size=10, top_k=4, batchwise_prompt=True, prompt_key_init='uniform',):
#         super().__init__()

#         self.length = length
#         self.embed_dim = embed_dim
#         self.embed_dim_key = embed_dim_key
#         self.prompt_pool = prompt_pool
#         self.embedding_key = embedding_key
#         self.prompt_init = prompt_init
#         self.prompt_key = prompt_key
#         self.pool_size = pool_size
#         self.top_k = top_k
#         self.batchwise_prompt = batchwise_prompt

#         if self.prompt_pool:
#             prompt_pool_shape = (pool_size, length, embed_dim)
#             if prompt_init == 'zero':
#                 self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#             elif prompt_init == 'uniform':
#                 self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
#                 nn.init.uniform_(self.prompt, -1, 1)
        
#         # if using learnable prompt keys
#         if prompt_key:
#             key_shape = (pool_size, embed_dim_key)
#             if prompt_key_init == 'zero':
#                 self.prompt_key = nn.Parameter(torch.zeros(key_shape))
#             elif prompt_key_init == 'uniform':
#                 self.prompt_key = nn.Parameter(torch.randn(key_shape))
#                 nn.init.uniform_(self.prompt_key, -1, 1)
#         else:
#             # else use mean of prompt as key
#             # only compatible with prompt, not prefix
#             prompt_mean = torch.mean(self.prompt, dim=1)
#             self.prompt_key = prompt_mean
    
#     def l2_normalize(self, x, dim=None, epsilon=1e-12):
#         """Normalizes a given vector or matrix."""
#         square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
#         x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
#         return x * x_inv_norm
    
#     def forward(self, x_embed, prompt_mask=None, cls_features=None):
#         out = dict()
#         if self.prompt_pool:
#             if self.embedding_key == 'mean':
#                 x_embed_mean = torch.mean(x_embed, dim=1)
#             elif self.embedding_key == 'max':
#                 x_embed_mean = torch.max(x_embed, dim=1)[0]
#             elif self.embedding_key == 'mean_max':
#                 x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
#             elif self.embedding_key == 'cls':
#                 if cls_features is None:
#                     x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
#                 else:
#                     x_embed_mean = cls_features
#             else:
#                 raise NotImplementedError("Not supported way of calculating embedding keys!")

#             prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
#             x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

#             similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
#             if prompt_mask is None:
#                 _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
#                 if self.batchwise_prompt:
#                     prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
#                     # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
#                     # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
#                     # Unless dimension is specified, this will be flattend if it is not already 1D.
#                     if prompt_id.shape[0] < self.pool_size:
#                         prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
#                         id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
#                     _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
#                     major_prompt_id = prompt_id[major_idx] # top_k
#                     # expand to batch
#                     idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
#             else:
#                 idx = prompt_mask # B, top_k

#             batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
#             batch_size, top_k, length, c = batched_prompt_raw.shape
#             batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

#             # out['prompt_idx'] = idx

#             # # Debugging, return sim as well
#             # out['prompt_norm'] = prompt_norm
#             # out['x_embed_norm'] = x_embed_norm
#             # out['similarity'] = similarity

#             # # Put pull_constraint loss calculation inside
#             # batched_key_norm = prompt_norm[idx] # B, top_k, C
#             # out['selected_key'] = batched_key_norm
#             # x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
#             # sim = batched_key_norm * x_embed_norm # B, top_k, C
#             # reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

#             # out['reduce_sim'] = reduce_sim

        
#         # The input with the prompt concatenated to the front. [B, prompt+token, C]
#         # out['total_prompt_len'] = batched_prompt.shape[1]
#         # out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

#         return batched_prompt

# zero_value = 1e-8
# class RepZeroConv2d(nn.Conv2d):
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding = 0,
#                  dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, zero_value=zero_value) -> None:
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
#         self.scaling = nn.parameter.Parameter(torch.ones(1) * vis_scale)
#         nn.init.constant_(self.weight, val=zero_value)
#         if self.bias is not None:
#             nn.init.constant_(self.bias, val=zero_value)
        
#         self.freeze_conv = nn.Conv2d(in_channels, out_channels,
#                                      kernel_size, stride, padding,
#                                      dilation, groups, bias,
#                                      padding_mode, device, dtype)
#         nn.init.constant_(self.freeze_conv.weight, val=0.0)
#         if self.bias is not None:
#             nn.init.constant_(self.freeze_conv.bias, val=0.0)
        
#         # self.zero_inter_loss = torch.nn.L1Loss(reduction='mean')
#         # self.zero_inter_loss = torch.nn.MSELoss(reduction='mean')
#         self.zero_inter_loss = torch.nn.SmoothL1Loss(reduction='mean')

#     def forward(self, input: Tensor) -> Tensor:
#         if self.training:
#             branch_output = self.scaling * super().forward(input)
#             output = branch_output + self.freeze_conv(input)
#             return output, \
#                 self.zero_inter_loss(branch_output, torch.zeros_like(branch_output)) + \
#                     self.zero_inter_loss(output, torch.zeros_like(output))
#         else:
#             return self.freeze_conv(input), torch.zeros(1).to(input)

#     def __rep__(self):
#         self.freeze_conv.weight.data = self.weight.data  * self.scaling + self.freeze_conv.weight.data
#         self.freeze_conv.bias.data = self.bias.data  * self.scaling + self.freeze_conv.bias.data
#         self.scaling = nn.parameter.Parameter(torch.ones(1).to(self.weight.data) *vis_scale)
#         nn.init.constant_(self.weight, val=zero_value)
#         if self.bias is not None:
#             nn.init.constant_(self.bias, val=zero_value)
            