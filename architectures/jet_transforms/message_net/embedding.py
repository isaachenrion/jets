import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dim = config.data_dim
        self.state_dim = config.state_dim

    def forward(self, x):
        pass

class FullyConnectedEmbedding(Embedding):
    def __init__(self, *args):
        super().__init__(*args)
        if self.config.ndim == 2:
            bn = nn.BatchNorm1d
        elif self.config.ndim == 3:
            bn = nn.BatchNorm2d
        self.net = nn.Sequential(
            nn.BatchNorm1d(self.data_dim),
            nn.Linear(self.data_dim, self.state_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.state_dim),
            nn.Linear(self.state_dim, self.state_dim)
        )

    def forward(self, x):
        bs, gd, dd = (s for s in x.size())
        x = x.view(-1, dd)
        x = self.net(x)
        x = x.view(bs, gd, -1)
        return x

class Constant(Embedding):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        return x

def make_embedding(embedding_config):
    if embedding_config.function == 'constant':
        return Constant(embedding_config.config)
    elif embedding_config.function == 'fully_connected':
        return FullyConnectedEmbedding(embedding_config.config)
    else:
        raise ValueError("Unsupported embedding function! ({})".format(embedding_config.function))
