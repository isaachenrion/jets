import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn_utils import AnyBatchGRUCell

class VertexUpdate(nn.Module):
    def __init__(self, message_dim, hidden_dim, vertex_state_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.vertex_state_dim = vertex_state_dim
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
    def __init__(self, message_dim, hidden_dim, vertex_state_dim=0):
        super().__init__(message_dim, hidden_dim, vertex_state_dim)
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
