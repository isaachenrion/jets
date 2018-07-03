import torch
import torch.nn as nn
import torch.nn.functional as F

def dot(a, b):
    """Compute the dot product between pairs of vectors in 3D tensors.

    Args
    ----
    a: tensor of size (B, M, D)
    b: tensor of size (B, N, D)

    Returns
    -------
    c: tensor of size (B, M, N)
        c[i,j,k] = dot(a[i,j], b[i,k])
    """
    return a.bmm(b.transpose(1, 2))

class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value, adj=None):
        ''' Input:
            query vectors q_1, ..., q_m
            key vectors k_1, .., k_n
            value vectors v_1, ..., v_n

            all of equal dimension and batch size.

            also binary adjacency matrix adj

            Compute the attention weights alpha_ij as follows:

            score_ij = (q_j)^T k_i
            alpha_ij = exp(score_ij) / sum_k exp(score_ik)

            Then apply the attention weights to v_1, ..., v_n as follows:

            output_ij = sum_k (alpha_ik * v_kj)
        '''
        bsk, n_keys, dim_key = key.size()
        bsq, n_queries, dim_query = query.size()
        bsv, n_queries, dim_value = value.size()

        try:
            assert bsq == bsk == bsv
            batch_size = bsk
        except AssertionError as e:
            logging.debug(
                'Mismatch: \
                query batch size = {}, \
                key batch size = {}, \
                value batch size = {} \
                but should be equal'.format(bsq, bsk, bsv))
            raise e
        try:
            assert dim_key == dim_query == dim_value
        except AssertionError as e:
            logging.debug(
                'Mismatch: \
                query data dimension = {}, \
                key data dimension = {}, \
                value data dimension = {} \
                but should be equal'.format(dim_query, dim_key, dim_value))
            raise e

        s = dot(query, key)

        dimensions  = torch.tensor([key.size()[1]]).view(1, 1, 1).expand_as(s).float()
        scaling_factor = torch.sqrt(1 / dimensions)

        if adj is not None:
            zero_vec = -9e15*torch.ones_like(s)
            s = torch.where(adj > 0, s, zero_vec)

        alpha = F.softmax(s / scaling_factor, dim=2)
        alpha = self.dropout(alpha)
        output = torch.bmm(alpha, value)

        return output, alpha


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(dropout)
        self.proj = nn.Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.w_qs)
        nn.init.xavier_normal_(self.w_ks)
        nn.init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, adj=None):

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        if adj is not None:
            adj = adj.repeat(n_head, 1, 1)
        outputs, attns = self.attention(q_s, k_s, v_s, adj)
        attns = attns.view(mb_size, n_head, len_q, -1)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return outputs, attns
