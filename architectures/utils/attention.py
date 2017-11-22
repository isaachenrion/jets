import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, context, query, target):
        ''' Input:
            context vectors c_1, .., c_n
            query vectors q_1, ..., q_m
            target vectors t_1, ..., t_n

            all of equal dimension and batch size.

            Compute the attention weights alpha_ij as follows:

            score_ij = (q_j)^T c_i
            alpha_ij = exp(score_ij) / sum_k exp(score_ik)

            Then apply the attention weights to t_1, ..., t_n as follows:

            output_ij = sum_k (alpha_ik * t_kj)
        '''
        bsc, n_contexts, dim_context = context.size()
        bsq, n_queries, dim_query = query.size()
        bst, n_queries, dim_target = target.size()

        try:
            assert bsq == bsc == bst
            batch_size = bsc
        except AssertionError as e:
            logging.debug(
                'Mismatch: \
                query batch size = {}, \
                context batch size = {}, \
                target batch size = {} \
                but should be equal'.format(bsq, bsc, bst))
            raise e
        try:
            assert dim_context == dim_query == dim_target
        except AssertionError as e:
            logging.debug(
                'Mismatch: \
                query data dimension = {}, \
                context data dimension = {}, \
                target data dimension = {} \
                but should be equal'.format(dim_query, dim_context, dim_target))
            raise e

        s = dot(query, context)
        alpha = F.softmax(s)
        output = torch.bmm(alpha, target)
        return output, alpha
