import torch
import torch.nn as nn
from ...utils import Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim_query, dim_value, dim_hidden):
        super().__init__()
        #self.query_weights_list = nn.Linear(n_heads * dim_hidden, n_heads * dim_query, bias=False)
        #self.key_weights_list = nn.Linear(n_heads * dim_hidden, n_heads * dim_query, bias=False)
        #self.value_weights_list = nn.Linear(n_heads * dim_hidden, n_heads * dim_value, bias=False)
        self.wqs = nn.ModuleList([nn.Linear(dim_hidden, dim_query, bias=False) for _ in range(n_heads)])
        self.wks = nn.ModuleList([nn.Linear(dim_hidden, dim_query, bias=False) for _ in range(n_heads)])
        self.wvs = nn.ModuleList([nn.Linear(dim_hidden, dim_value, bias=False) for _ in range(n_heads)])
        self.wo = nn.Linear(n_heads * dim_value, dim_hidden, bias=False)
        self.n_heads = n_heads
        self.attention = Attention()

    def forward(self, q, k, v):
        #hq = self.query_weights_list(q.repeat(1, self.n_heads))
        #hk = self.key_weights_list(k.repeat(1, self.n_heads))
        #hv = self.value_weights_list(v.repeat(1, self.n_heads))
        heads = []
        for wq, wk, wv in zip(self.wqs, self.wks, self.wvs):
            hq = wq(q)
            hk = wk(k)
            hv = wv(v)
            head, _ = self.attention(hq, hk, hv)
            heads.append(head)
        heads = torch.cat(heads, 2)
        output = self.wo(heads)
        return output
