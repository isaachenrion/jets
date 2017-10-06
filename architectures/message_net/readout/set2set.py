import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Set2Vec(nn.Module):
    def __init__(self, input_dim, output_dim, memory_dim):
        super().__init__()
        #self.input_dim, self.output_dim, self.memory_dim = input_dim, output_dim, memory_dim

        self.embedding = nn.Sequential(nn.Linear(input_dim, memory_dim), nn.ReLU(), nn.Linear(memory_dim, memory_dim))
        self.process = ProcessBlock(2 * memory_dim)
        self.write = nn.Linear(2 * memory_dim, output_dim)

    def forward(self, x):
        ''' x has shape (batch_size, seq_length, feature_dim) '''
        # embed each element of sequence into a memory vector
        m = self.embedding(x) # m has shape (bs, L, mem_dim)
        # process the memories with content-based attention
        q = Variable(torch.zeros(m.size()[0], 2 * m.size()[2]))
        if torch.cuda.is_available(): q = q.cuda()
        for t in range(x.size()[1]):
            q = self.process(q, m)
        # readout from the final hidden state
        output = self.write(q)
        return output


class ProcessBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.recurrent = NoInputGRUCell(input_dim)

    def lookup(self, q, m):
        return torch.matmul(m, q.unsqueeze(2)).squeeze(2)

    def forward(self, q, m):
        q_hat, _ = self.recurrent(q).chunk(2, 1)
        e = self.lookup(q_hat, m)
        a = F.softmax(e)
        r = torch.sum(a.unsqueeze(2) * m, 1)
        q = torch.cat([q_hat, r], 1)
        return q

class NoInputGRUCell(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 3 * hidden_dim)

    def forward(self, h):
        r_, z_, n_ = self.linear(h).chunk(3, 1)
        r = F.sigmoid(r_)
        z = F.sigmoid(z_)
        n = F.tanh(r * n_)
        h = (1 - z) * n + z * h
        return h
