import torch
from torch import nn


def log_sum_exp(smat):
    pass

class BiLSTM_CRF(nn.Module):
    def neg_log_likehood(self,words,tags):
        frames = self._get_lstm_feature(words)
        gold_score = self._score_sentence((frames,tags))
        forward_score = self._forward_alg(frames)
        return forward_score - gold_score

    def _score_sentence(self, frames, tags):
        tags_tensor = self._to_tensor([START_TAG] + tags,self.tag2ix)
        score =torch.zeros(1)
        for i,frame in enumerate(frames):
            score += self.transitions[tags_tensor[i],tags_tensor[i+1] + frame[tags_tensor[i+1]]]
        return score + self.transitions[tags_tensor[-1],self.tag2ix[END_TAG]]



    def _forward_alg(self, frames):
        pass

    def _viterbi_decode(self, frames):
        pass

    def forward(self, words):
        pass



