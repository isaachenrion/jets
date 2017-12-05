from .stacked_mpnn import StackedMPNNTransform
from .attention_pooling import AttentionPooling
from .attention_pooling import RecurrentAttentionPooling

POOLINGS = {
    'attn': (0, AttentionPooling),
    'rec-attn': (1, RecurrentAttentionPooling)
}
