import torch
import torch.nn as nn
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v):
        # [batch_size, self.n_head, length, d_tensor]
        _, _, _, d_k = k.size()
        # (q_length, d_q) x (d_k, k_length) -> (q_length, k_length), where d_q = d_k
        qk = q @ k.transpose(2, 3)
        score = qk / math.sqrt(d_k)
        score = self.softmax(score)
        
        # (q_length, k_length) x (k_length, d_v) -> (q_length, d_v)
        res = score @ v
        
        return res
