import torch
import torch.nn as nn
from torch.nn import init
from ...utils import Attention
from ...utils import BottleLinear as Linear

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_model, dropout=False, **kwargs):
        super().__init__()
        #self.query_weights_list = nn.Linear(n_heads * dim_hidden, n_heads * dim_query, bias=False)
        #self.key_weights_list = nn.Linear(n_heads * dim_hidden, n_heads * dim_query, bias=False)
        #self.value_weights_list = nn.Linear(n_heads * dim_hidden, n_heads * dim_value, bias=False)
        #self.wqs = nn.ModuleList([nn.Linear(dim_hidden, dim_query, bias=False) for _ in range(n_heads)])
        #self.wks = nn.ModuleList([nn.Linear(dim_hidden, dim_query, bias=False) for _ in range(n_heads)])
        #self.wvs = nn.ModuleList([nn.Linear(dim_hidden, dim_value, bias=False) for _ in range(n_heads)])
        #self.wo = nn.Linear(n_heads * dim_value, dim_hidden, bias=False)
        self.wqs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.wks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.wvs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        init.xavier_normal(self.wqs)
        init.xavier_normal(self.wks)
        init.xavier_normal(self.wvs)

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.attention = Attention()
        self.proj = Linear(n_head*d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.wqs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.wks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.wvs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        #heads = []
        #for wq, wk, wv in zip(self.wqs, self.wks, self.wvs):
        #    hq = wq(q)
        #    hk = wk(k)
        #    hv = wv(v)
        #    head, _ = self.attention(hq, hk, hv)
        #    heads.append(head)
        #heads = torch.cat(heads, 2)
        #output = self.wo(heads)
        return outputs
