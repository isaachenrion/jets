import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def dot(a, b):
    """Compute the dot product between pairs of vectors in 3D Variables.

    Args
    ----
    a: Variable of size (B, M, D)
    b: Variable of size (B, N, D)

    Returns
    -------
    c: Variable of size (B, M, N)
        c[i,j,k] = dot(a[i,j], b[i,k])
    """
    return a.bmm(b.transpose(1, 2))

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, dimensions=None):
        ''' Input:
            query vectors q_1, ..., q_m
            key vectors k_1, .., k_n
            value vectors v_1, ..., v_n

            all of equal dimension and batch size.

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
        if dimensions is None:
            dimensions  = Variable(torch.FloatTensor([key.size()[1]]).view(1, 1, 1).expand_as(s))
        scaling_factor = torch.sqrt(1 / dimensions)
        alpha = F.softmax(s / scaling_factor)
        output = torch.bmm(alpha, value)
        return output, alpha
