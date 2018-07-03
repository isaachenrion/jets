import os
import logging
import math
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.architectures.nmp.message_passing.vertex_update import GRUUpdate
from src.architectures.utils.attention import MultiHeadAttention

from src.admin.utils import memory_snapshot

from ..ProteinModel import ProteinModel

class GraphAttention(ProteinModel):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        n_head=None,
        **kwargs
        ):
        super().__init__()

        self.embedding = nn.Linear(features, hidden)
        self.graph_attention_layers = nn.ModuleList([MultiHeadAttention(n_head, hidden, hidden, hidden) for _ in range(iters)])

    def forward(self, x, mask, **kwargs):
        x = self.embedding(x)
        for GAL in self.graph_attention_layers:
            x, _ = GAL(x, x, x, adj=None)
        contact_map = F.sigmoid(torch.bmm(x, x.transpose(1,2))) * mask
        return contact_map
