
import torch
import torch.nn as nn
import torch.nn.functional as F

from .....architectures.embedding import construct_embedding

class Message(nn.Module):
    ''' Base class implementing neural message function for MPNNs.
    '''
    def __init__(self, hidden_dim, message_dim, edge_dim):
        super().__init__()
        self.vertex_dim = hidden_dim
        self.edge_dim = edge_dim
        self.message_dim = message_dim

    def forward(self, *args):
        if self.edge_dim == 0:
            return self._forward_without_edge(*args)
        else:
            return self._forward_with_edge(*args)

    def _forward_with_edge(self, vertices, edges):
        raise NotImplementedError

    def _forward_without_edge(self, vertices):
        raise NotImplementedError

class DTNNMessage(Message):
    def __init__(self, *args):
        super().__init__(*args)
        self.vertex_wx_plus_b = nn.Linear(self.vertex_dim, self.message_dim)
        if self.edge_dim > 0:
            self.edge_wx_plus_b = nn.Linear(self.edge_dim, self.message_dim)
            self.combo_wx = nn.Linear(self.message_dim, self.message_dim, bias=False)

    def _forward_with_edge(self, vertices, edges):

        message = self.combo_wx(self.vertex_wx_plus_b(vertices) * self.edge_wx_plus_b(edges))
        return message

    def _forward_without_edge(self, vertices):
        message = self.vertex_wx_plus_b(vertices)
        return message

class SimpleMessage(Message):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.embedding = construct_embedding('simple', self.vertex_dim, self.message_dim, **kwargs)

    def _forward_without_edge(self, vertices):
        message = self.embedding(vertices)
        return message

class ReLUMessage(Message):
    def __init__(self,*args):
        super().__init__(*args)
        self.embedding = construct_embedding('simple', self.vertex_dim, self.message_dim, act='relu')

    def _forward_without_edge(self, vertices):
        message = self.embedding(vertices)
        return message

class FullyConnectedMessage(Message):
    def __init__(self, *args):
        super().__init__(*args)
        self.net = nn.Sequential(
            nn.Linear(self.vertex_dim + self.edge_dim, self.message_dim),
            nn.ReLU()
            )

    def _forward_with_edge(self, vertices, edges):
        return self.net(torch.cat([vertices, edges], -1))

    def _forward_without_edge(self, vertices):
        return self.net(vertices)

class EdgeMatrixMessage(Message):
    def __init__(self, *args):
        super().__init__(*args)
        self.edge_net = nn.Linear(self.edge_dim, self.message_dim * self.vertex_dim)

    def _forward_with_edge(self, vertices, edges):

        if vertices.dim() == 2:
            message = torch.matmul(
                self.edge_net(edges).view(edges.size()[0], self.message_dim, self.vertex_dim),
                vertices.unsqueeze(-1)).squeeze(-1)

        elif vertices.dim() == 3:
            message = torch.matmul(
                self.edge_net(edges).view(edges.size()[0], edges.size()[1], self.message_dim, self.vertex_dim),
                vertices.unsqueeze(-1)).squeeze(-1)

        return message

class Constant(Message):
    def __init__(self, config):
        super().__init__(config)


    def _forward_without_edge(self, vertices):
        return vertices

    def _forward_with_edge(self, vertices, edges):
        message = torch.cat([edges, vertices], -1)
        return message
