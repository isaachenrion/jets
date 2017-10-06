from .mpnn import BaseMPNN
from .embedding import make_embedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GCN1(BaseMPNN):
    def __init__(self, config):
        super().__init__(config)
        self.activation = F.tanh
        self.embedding = make_embedding(config.embedding)

    def forward(self, vertices, dads):
        h = Variable(torch.zeros(vertices.size()[0], vertices.size()[1], self.config.message.config.hidden_dim))
        if torch.cuda.is_available(): h = h.cuda()
        s = self.embedding(vertices) # vertex states
        for i in range(self.n_iters):
            h = self.message_passing(h, s, dads)
        out = self.readout(h)
        return out

    def message_passing(self, h, s, dads):
        message = self.activation(torch.matmul(dads, self.message(h)))
        h = self.vertex_update(h, message, s)
        return h

class GCN2(BaseMPNN):
    def __init__(self, config):
        super().__init__(config)
        self.activation = F.tanh
        self.embedding = make_embedding(config.embedding)

    def forward(self, vertices, dads):
        h = self.embedding(vertices) # init hiddens with embedding
        for i in range(self.n_iters):
            h = self.message_passing(h, dads)
        out = self.readout(h)
        return out

    def message_passing(self, h, dads):
        message = self.activation(torch.matmul(dads, self.message(h)))
        h = self.vertex_update(h, message)
        return h

class VCN(BaseMPNN):
    def __init__(self, config):
        super().__init__(config)
        self.activation = F.tanh
        self.embedding = make_embedding(config.embedding)

    def forward(self, vertices, dads):
        h = Variable(torch.zeros(vertices.size()[0], vertices.size()[1] + 1, self.config.message.config.hidden_dim))
        if torch.cuda.is_available(): h = h.cuda()

        s = self.embedding(vertices) # vertex states
        for i in range(self.n_iters):
            h = self.message_passing(h, s, dads)
        out = self.readout(h)
        return out

    def message_passing(self, h, s, dads):
        raw_message = self.message(h)
        global_vertex = self.activation(torch.mean(raw_message, -1))
        message = self.activation(torch.matmul(dads, raw_message))
        h = self.vertex_update(h, message, s)
        return h, global_vertex


def make_gcn(config):
    if config.gcn_type == 'gcn1':
        return GCN1(config)
    elif config.gcn_type == 'gcn2':
        return GCN2(config)
