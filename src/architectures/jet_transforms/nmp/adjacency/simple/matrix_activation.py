import torch
import torch.nn.functional as F


def padded_matrix_softmax(matrix, mask):
    '''
    Inputs:
        matrix <- (batch_size) * M * M tensor that has been padded
        mask <- (batch_size * M * M) with zeros to mask out the fictitious nodes
    Output:
        S <- (batch_size) * M * M tensor, where S[n, i] is a probability distribution over the
            values 1, ..., M. The softmax is taken over each row of the
            matrix, and the padded values have been assigned probability 0.
    '''
    S = F.softmax(-matrix.transpose(0, -1)).transpose(0, -1)
    if mask is not None:
        S = S * mask
    Z = S.sum(2, keepdim=True) + 1e-10
    S = S / Z
    return S

def masked_function(fn):
    def masked(matrix, mask):
        return fn(matrix) * mask
    return masked

def no_mask_softmax(matrix, mask):
    return F.softmax(-matrix.transpose(0, -1)).transpose(0, -1)

MATRIX_ACTIVATIONS = {
    'mask': masked_function(lambda x: x),
    'soft': padded_matrix_softmax,
    'sigmoid': masked_function(F.sigmoid),
    'exp': masked_function(lambda x: torch.exp(-x)),
    'tanh': masked_function(F.tanh),
    'no_mask_softmax': no_mask_softmax
}
