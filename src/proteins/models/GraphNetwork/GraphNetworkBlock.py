import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphNetworkBlock(nn.Module):
    def __init__(self, graph_dim, node_dim, edge_dim, graph_dim_out=None, node_dim_out=None, edge_dim_out=None):
        super().__init__()

        if graph_dim_out is None:
            graph_dim_out = graph_dim
        if node_dim_out is None:
            node_dim_out = node_dim
        if edge_dim_out is None:
            edge_dim_out = edge_dim

        self.f_A_nn = nn.Linear(edge_dim + 2 * node_dim + graph_dim, edge_dim_out)
        self.f_v_nn = nn.Linear(edge_dim_out + node_dim + graph_dim, node_dim_out)
        self.f_u_nn = nn.Linear(edge_dim_out + node_dim_out + graph_dim, graph_dim_out)

    def rho_A_v(self, A_prime):
        return A_prime.mean(1)

    def rho_A_u(self, A_prime):
        return A_prime.mean(1).mean(1)

    def rho_v_u(self, V_prime):
        return V_prime.mean(1)

    def f_A(self, A, V, u):
        bs, n_nodes, node_dim = V.shape
        V = V.unsqueeze(1).expand(bs, n_nodes, n_nodes, *V.shape[2:])
        u = u.unsqueeze(1).unsqueeze(1).expand(bs, n_nodes, n_nodes, *u.shape[1:])
        x = torch.cat([A,V,V.transpose(1,2),u], -1)
        x = self.f_A_nn(x)
        x = F.relu(x)
        return x

    def f_u(self, A_agg, v_agg, u):
        x = torch.cat([A_agg,v_agg,u], -1)
        x = self.f_u_nn(x)
        x = F.relu(x)
        return x

    def f_v(self, A_agg_by_node, V, u):
        bs, n_nodes, node_dim = V.shape
        u = u.unsqueeze(1).expand(bs, n_nodes, *u.shape[1:])
        x = torch.cat([A_agg_by_node,V,u], -1)
        x = self.f_v_nn(x)
        x = F.relu(x)
        return x

    def forward(self, u, V, A):
        # compute updated edge attributes
        A_prime = self.f_A(A, V, u)

        # aggregate edge attributes per node
        A_agg_by_node = self.rho_A_v(A_prime)

        # compute updated node attributes
        V_prime = self.f_v(A_agg_by_node, V, u)

        # aggregate edge attributes globally
        A_agg = self.rho_A_u(A_prime)

        # aggregate node attributes globally
        v_agg = self.rho_v_u(V_prime)

        # compute updated global attribute
        u_prime = self.f_u(A_agg, v_agg, u)

        return (u_prime, V_prime, A_prime)
