import json

import torch
import numpy as np
import warnings


class preprocessing():
    def __init__(self, hidden_dim, word2idx, max_positions):
        self.max_positions = max_positions + 2
        self.hidden_dim = hidden_dim
        self.word2idx = word2idx
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4
        self.num_index = 5
        self.positional_encoding = self.init_positional_encoding()



    def init_positional_encoding(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_positions)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc ** 2, axis=1, keepdims=True))
        position_enc = position_enc / (denominator + 1e-8)
        return position_enc
        # 输出【250，384】：【max_positions,hidden_dim】

    def tokenize(self, text_or_label, dict):
        return [dict.get(i, self.unk_index) for i in text_or_label]

    def add_cls_sep(self, text_tokens):
        return [self.cls_index] + text_tokens + [self.sep_index]

    def add_cls_sep_padding(self, tokens):
        return [self.pad_index] + tokens + [self.pad_index]

    def __call__(self, text_list, max_seq_len):
        text_list_len = [len(i) for i in text_list]
        # if max(text_list_len) > self.max_positions -2:
        #     raise AssertionError(
        #         "max_seq_len exceeds the maximum length of positional encoding! 输入的最大文本长度{}大于最大位置嵌入允许的长度{}!".format(
        #             max(text_list_len), self.max_positions-2))

        if max(text_list_len) > max_seq_len:
            warnings.warn(
                "maximum length of input texts exceeds \"max_seq_len\"! exceeded length will be cut off! "
                "输入的最大文本长度大于指定最大长度, 多余的部分将会被剪切! "
            )

        batch_max_seq_len = max_seq_len + 2
        # tokenize
        texts_tokens = [self.tokenize(i, self.word2idx) for i in text_list]
        # add cls, sep
        texts_tokens = [self.add_cls_sep(i) for i in texts_tokens]
        # padding
        texts_tokens = [torch.tensor(i) for i in texts_tokens]
        texts_tokens = torch.nn.utils.rnn.pad_sequence(texts_tokens, batch_first=True)
        positional_enc = \
            torch.from_numpy(self.positional_encoding[:batch_max_seq_len]).type(torch.FloatTensor)
        return texts_tokens, positional_enc



# with open("../corpus/bert_word2idx_extend.json", "r", encoding="utf-8") as f:
#     word2idx = json.load(f)
# a = preprocessing(hidden_dim=384, word2idx=word2idx, max_positions=250)
# c = a.init_positional_encoding()
# print(c)
# text_list=['去上海一般都住这家，地段还不错，交通也比较方便，房间很大也比较干净，就是江景房有点差（以前住过，很小），一般标房就可以了，套房也很不错（有厨房，够大），每件房都有冰箱，很好，就是宽带不是免费的，一天好像要80吧，太贵了']
# max_seq_len = 250
# b = a(text_list, max_seq_len)
# b =b[0]
# print(b)
# print(b.shape)
