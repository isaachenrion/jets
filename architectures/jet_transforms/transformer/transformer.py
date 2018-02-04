import torch
import torch.nn as nn

from .multihead_attention import MultiHeadAttention
from ...utils.layer_norm import LayerNorm

class Transformer(nn.Module):
    def __init__(self,
        hidden=None,
        n_heads=None,
        n_layers=None
        ):
        super().__init__()
        self.transformer_layers = nn.ModuleList([SelfAttentionLayer(hidden, n_heads) for _ in range(n_layers)])

    def forward(self, x, **kwargs):
        '''
        x has dimension (B, N, D) where
            B = batch size
            N = number of nodes
            D = model dimension
        '''
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(n_heads, 64, 64, hidden)
        self.ln1 = LayerNorm(hidden)
        self.ff = nn.Sequential(
                    nn.Linear(hidden, 4 * hidden),
                    nn.ReLU(),
                    nn.Linear(4 * hidden, hidden)
                    )
        self.ln2 = LayerNorm(hidden)

    def forward(self, x):
        x = x + self.multihead_attention(x, x, x)
        x = self.ln1(x)
        x = x + self.ff(x)
        x = self.ln2(x)
        return x
