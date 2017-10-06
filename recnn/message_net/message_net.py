import torch
import torch.nn as nn
import torch.nn.functional as F

from ..batching import pad_batch, batch

class MPNNTransform(nn.Module):
    def __init__(self, n_features, n_hidden, n_iters):
        super().__init__()
        self.n_hidden = n_hidden

        self.activation = F.tanh
        self.embedding = Embedding()
        self.vertex_update = VertexUpdate()
        self.message = Message()
        self.readout = Readout()

    def forward(self, jets):
        jets = pad_batch(jets)

        h = Variable(torch.zeros(jets.size()[0], jets.size()[1], self.n_hidden))
        if torch.cuda.is_available(): h = h.cuda()

        h = self.embedding(jets) # vertex states
        for i in range(self.n_iters):
            h = self.message_passing(h, s, dads)
        out = self.readout(h)
        return out

    def message_passing(self, h, s, dads):
        message = self.activation(torch.matmul(dads, self.message(h)))
        h = self.vertex_update(h, message, s)
        return h
