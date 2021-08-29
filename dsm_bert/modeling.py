from torch import nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel
import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)
from dsm_bert import config


# 基于nn.Module构建的模型存在无法将模型的加载标准化的问题
class dsm_ernie(nn.Module):
    """ 构建双塔 ernie匹配模型,基于nn.Module
        所谓双塔是指两个输入源经过两个模型进行编码后进行cosine相似度计算
        匹配是指计算的相似度于标签可以作为一个分类任务去处理，这里是用的一个简单的线性函数来处理的
    """

    def __init__(self, model):
        super().__init__()
        # ernie模型
        self.bert = model
        # cosine相似度计算
        self.sim = F.cosine_similarity
        # 全连接层
        self.fc = nn.Linear(config.max_length, 2)

    def forward(self, query, title):
        # 将query和title的tokenizer后的结果以解码的方式输入模型，并获取其中的last_hidden_state作为embedding
        out_q = self.bert(**query)
        out_t = self.bert(**title)
        # 形状为（batch_size,seq_length,embedding_dim)
        query_embed = out_q.last_hidden_state
        title_embed = out_t.last_hidden_state

        # 计算两个embedding层的余弦相似度
        cosine = self.sim(query_embed, title_embed, dim=-1)
        # 对计算出来的余弦相似度进行线性映射，并做softmax归一化得到概率
        prob = F.softmax(self.fc(cosine), dim=-1)
        return prob


class Dsm_Bert(BertPreTrainedModel):
    """ 参考https://github.com/huggingface/transformers/issues/5816编写自定义模块
        构建双塔 ernie匹配模型,基于BertPreTrainedModel
        所谓双塔是指两个输入源经过两个模型进行编码后进行cosine相似度计算
        匹配是指计算的相似度于标签可以作为一个分类任务去处理，这里是用的一个简单的线性函数来处理的
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        # self.init_weights()
        self.l = nn.Linear(config.max_length, 2)

    def forward(self, query, title):
        # 获取query和title的out
        out_k = self.bert(**query)
        out_t = self.bert(**title)
        # 将
        embed_q = out_k.last_hidden_state
        embed_t = out_t.last_hidden_state

        cosine = F.cosine_similarity(embed_q, embed_t, dim=-1)
        prob = F.softmax(self.l(cosine), dim=-1)

        return prob
