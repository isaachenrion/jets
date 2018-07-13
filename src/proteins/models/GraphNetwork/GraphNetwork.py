import torch
import torch.nn as nn

from .GraphNetworkBlock import GraphNetworkBlock
from .CombinationBlocks import DenseGraphNetworkBlock, ResidualGraphNetworkBlock


class GraphNetwork(nn.Module):
    '''
    Follows the encode-process-decode framework
    '''
    def __init__(self,
                 graph_dim_in,
                 node_dim_in,
                 edge_dim_in,
                 graph_dim_hidden,
                 node_dim_hidden,
                 edge_dim_hidden,
                 graph_dim_out,
                 node_dim_out,
                 edge_dim_out,
                 n_process_layers
                ):

        super().__init__()

        self.graph_dim_in, \
        self.node_dim_in,\
        self.edge_dim_in,\
        self.graph_dim_hidden,\
        self.node_dim_hidden,\
        self.edge_dim_hidden,\
        self.graph_dim_out,\
        self.node_dim_out,\
        self.edge_dim_out,\
        self.n_process_layers = \
                                graph_dim_in,\
                                node_dim_in,\
                                edge_dim_in,\
                                graph_dim_hidden,\
                                node_dim_hidden,\
                                edge_dim_hidden,\
                                graph_dim_out,\
                                node_dim_out,\
                                edge_dim_out,\
                                n_process_layers

        self.gn_encode = GraphNetworkBlock(
                        graph_dim_in,
                        node_dim_in,
                        edge_dim_in,
                        graph_dim_hidden,
                        node_dim_hidden,
                        edge_dim_hidden
                        )
        self.gn_process = ResidualGraphNetworkBlock(
                         graph_dim_hidden,
                         node_dim_hidden,
                         edge_dim_hidden,
                         n_process_layers
                         )
        self.gn_decode = GraphNetworkBlock(
                         graph_dim_hidden,
                         node_dim_hidden,
                         edge_dim_hidden,
                         graph_dim_out,
                         node_dim_out,
                         edge_dim_out
                        )

    def forward(self, u, V, A):
        u, V, A = self.gn_encode(u, V, A)
        u, V, A = self.gn_process(u, V, A)
        u, V, A = self.gn_decode(u, V, A)
        return u, V, A

class ProteinGraphNetwork(nn.Module):
    def __init__(self,
                 features=None,
                 hidden=None,
                 iters=None,
                 **kwargs
                ):

        super().__init__()
        self.gn = GraphNetwork(
                    graph_dim_in=1,
                    node_dim_in=features,
                    edge_dim_in=1,
                    graph_dim_hidden=hidden,
                    node_dim_hidden=hidden,
                    edge_dim_hidden=hidden,
                    graph_dim_out=hidden,
                    node_dim_out=hidden,
                    edge_dim_out=hidden,
                    n_process_layers=iters
        )
        self.f_A_out = nn.Linear(hidden, 1)

        self.threshold = nn.Parameter(torch.tensor([[1.0]]))

    def forward(self, x, mask, *args):
        V = x
        bs, n_nodes, _ = V.shape
        u = torch.zeros(bs, self.gn.graph_dim_in)
        V_ = V.unsqueeze(1).expand(bs, n_nodes, n_nodes, *V.shape[2:])
        A = (V_ - V_.transpose(1,2)).pow(2).sum(-1, keepdim=True).pow(0.5).repeat(1,1,1,self.gn.edge_dim_in)

        u, V, A = self.gn(u, V, A)
        A = self.f_A_out(A)

        return A.squeeze(-1)
