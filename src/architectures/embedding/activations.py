import torch.nn as nn

ACTIVATIONS = dict(
    relu=nn.ReLU,
    leakyrelu=nn.LeakyReLU,
    tanh=nn.Tanh,
    sigmoid=nn.Sigmoid,
    elu=nn.ELU,
    selu=nn.SELU,
)
