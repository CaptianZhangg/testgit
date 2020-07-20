from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import math
import sys
from io import open
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FUN = {"gelu":gelu, "relu":torch.nn.functional.relu}

class BertConfig(object):
    def __init__(self,
                 vocab_size, # 字典字数
                 hidden_size=384, # 隐藏层维度也就是字向量维度
                 num_hidden_layers=6, # transformer block 的个数
                 num_attention_heads=12, # 注意力机制"头"的个数
                 intermediate_size=384*4, # feedforward层线性映射的维度
                 hidden_act="gelu", # 激活函数
                 hidden_dropout_prob=0.4, # dropout的概率
                 attention_probs_dropout_prob=0.4,
                 max_position_embeddings=512*2,
                 type_vocab_size=256, # 用来做next sentence预测,
                 # 这里预留了256个分类, 其实我们目前用到的只有0和1
                 initializer_range=0.02 # 用来初始化模型参数的标准差
                 ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

class BertEmbeddings(nn.Module):
    def __init__(self,config):
        super(BertEmbeddings, self).__init__()
