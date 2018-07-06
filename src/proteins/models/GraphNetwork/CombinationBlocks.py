import torch
import torch.nn as nn
import torch.nn.functional as F

from .GraphNetworkBlock import GraphNetworkBlock

class DenseGraphNetworkBlock(nn.Module):
    '''
    DenseNet approach to Graph Networks.
    We have a number of GN layers in the block.
    At each step, we concatenate the output of the last GN layer with its input.
    This way, we have skip-connections between all layers.
    '''
    def __init__(self, graph_dim, node_dim, edge_dim, n_layers):
        super().__init__()
        self.gn_blocks = nn.ModuleList([GraphNetworkBlock(i * graph_dim, i * node_dim, i * edge_dim, graph_dim, node_dim, edge_dim) for i in range(1, n_layers+1)])

    def forward(self, u, V, A, *args):
        for gn_block in self.gn_blocks:
            u_, V_, A_ = gn_block(u, V, A)
            u = torch.cat([u, u_], -1)
            V = torch.cat([V, V_], -1)
            A = torch.cat([A, A_], -1)

        return u_, V_, A_

class ResidualGraphNetworkBlock(nn.Module):
    '''
    ResNet approach to Graph Networks.
    We have a number of GN layers in the block.
    At each step, we add the output of the last GN layer to its input.
    This way, we have residual connections.
    '''
    def __init__(self, graph_dim, node_dim, edge_dim, n_layers):
        super().__init__()
        self.gn_blocks = nn.ModuleList([GraphNetworkBlock(graph_dim, node_dim, edge_dim, graph_dim, node_dim, edge_dim) for i in range(1, n_layers+1)])

    def forward(self, u, V, A, *args):
        for gn_block in self.gn_blocks:
            u_, V_, A_ = gn_block(u, V, A)
            u = u + u_
            V = V + V_
            A = A + A_
        return u_, V_, A_
