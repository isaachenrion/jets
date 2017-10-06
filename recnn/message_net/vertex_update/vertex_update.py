import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import AnyBatchGRUCell

class VertexUpdate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.message_dim = config.message_dim
        self.vertex_state_dim = config.vertex_state_dim
        self.has_vertex_state = self.vertex_state_dim > 0

    def forward(self, *args):
        if self.has_vertex_state:
            return self._forward_with_vertex_state(*args)
        else:
            return self._forward_without_vertex_state(*args)

    def _forward_without_vertex(self, *args):
        raise Error("Subclasses of VertexUpdate must implement _forward_without_vertex")

    def _forward_with_vertex(self, *args):
        raise Error("Subclasses of VertexUpdate must implement _forward_with_vertex")


class GRUUpdate(VertexUpdate):
    def __init__(self, config):
        super().__init__(config)
        self.activation = F.tanh
        self.gru = AnyBatchGRUCell(self.message_dim + self.vertex_state_dim, self.hidden_dim)

    def _forward_with_vertex_state(self, h, message, s):
        if s is None:
            gru_input = message
        else:
            gru_input = torch.cat((message, s), 2)
        h = self.gru(gru_input, h)
        return h


    def _forward_without_vertex_state(self, h, message):
        h = self.gru(message, h)
        return h


def make_vertex_update(vertex_update_config):
    if vertex_update_config.function == 'gru':
        return GRUUpdate(vertex_update_config.config)
    else:
        raise ValueError("Unsupported vertex update function! ({})".format(vertex_update_config.function))
