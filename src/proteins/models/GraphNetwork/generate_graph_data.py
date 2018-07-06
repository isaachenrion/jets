import numpy as np
import torch

def generate_graph_data(graph_dim = 13, node_dim = 7, edge_dim = 11, n_nodes = 17, n_edges = None):
    if n_edges is None:
        n_edges = int(n_nodes * np.round(np.log(n_nodes)))

    v = torch.randn(n_nodes, node_dim)
    E_temp = torch.randperm(n_nodes ** 2)[:n_edges]
    s = E_temp % n_nodes
    r = (E_temp / n_nodes)
    adj_list = [torch.masked_select(s, r == i) for i in range(n_nodes)]

    e = torch.randn(n_edges, edge_dim)
    u = torch.randn(graph_dim)

    E = (e, s, r)
    V = v

    return u, V, E
