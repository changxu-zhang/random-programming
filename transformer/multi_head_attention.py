import torch
import torch.nn as nn  
from scale_dot_product import ScaleDotProductAttention
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        
        # linear projection
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bs, length, d = q.size()
         
        # 1. linear projection (dot product with weight matrices)
        q, k, v = self.q_linear(q), self.k_linear(k), self.v_linear(v)
        
        # 2. split tensor by number of heads
        # [batch_size, length, d_model] --> [batch_size, self.n_head, length, d_tensor]
        d_tensor = d // self.n_head
        q = q.view(bs, length, self.n_head, d_tensor).permute(0, 2, 1, 3)
        k = k.view(bs, length, self.n_head, d_tensor).permute(0, 2, 1, 3)
        v = v.view(bs, length, self.n_head, d_tensor).permute(0, 2, 1, 3) 
        
        # 3. Scaled Dot-Product Attention
        res = self.attention(q, k, v)
        
        return res
    
