import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphNetworkBlockUnbatched(nn.Module):
    def __init__(self, graph_dim, node_dim, edge_dim, graph_dim_out=None, node_dim_out=None, edge_dim_out=None):
        super().__init__()

        if graph_dim_out is None:
            graph_dim_out = graph_dim
        if node_dim_out is None:
            node_dim_out = node_dim
        if edge_dim_out is None:
            edge_dim_out = edge_dim

        self.f_e_nn = nn.Linear(edge_dim + 2 * node_dim + graph_dim, edge_dim_out)
        self.f_v_nn = nn.Linear(edge_dim_out + node_dim + graph_dim, node_dim_out)
        self.f_u_nn = nn.Linear(edge_dim_out + node_dim_out + graph_dim, graph_dim_out)

    def rho_e_v(self, E_prime_i):
        return E_prime_i.mean(0)

    def rho_e_u(self, E_prime):
        e_prime, s, r = E_prime
        return e_prime.mean(0)

    def rho_v_u(self, V_prime):
        return V_prime.mean(0)

    def f_e(self, e, v_s, v_r, u):
        u = u.expand(e.shape[0],*u.shape)
        x = torch.cat([e,v_s,v_r,u], 1)
        x = self.f_e_nn(x)
        x = F.relu(x)
        return x

    def f_u(self, e_agg, v_agg, u):
        x = torch.cat([e_agg,v_agg,u], 0)
        x = self.f_u_nn(x)
        x = F.relu(x)
        return x

    def f_v(self, e_agg_by_node, v, u):
        u = u.expand(v.shape[0],*u.shape)
        x = torch.cat([e_agg_by_node,v,u], 1)
        x = self.f_v_nn(x)
        x = F.relu(x)
        return x

    def forward(self, u, V, E):
        e, s, r = E

        v_s = torch.index_select(V, 0, s)
        v_r = torch.index_select(V, 0, r)

        # compute updated edge attributes
        e_prime = self.f_e(e, v_s, v_r, u)

        # aggregate edge attributes per node
        e_agg_by_node = []
        for i in range(V.shape[0]):
            indices = (r==i).nonzero()
            if len(indices) > 0:
                indices=indices[:,0].long()
                E_prime_i = torch.index_select(e_prime, 0, indices)
                e_agg_by_node.append(self.rho_e_v(E_prime_i))
            else:
                e_agg_by_node.append(torch.zeros(e_prime.shape[1]))
        e_agg_by_node = torch.stack(e_agg_by_node, 0)

        # compute updated node attributes
        V_prime = self.f_v(e_agg_by_node, V, u)

        E_prime = e_prime, s, r

        # aggregate edge attributes globally
        e_agg = self.rho_e_u(E_prime)

        # aggregate node attributes globally
        v_agg = self.rho_v_u(V_prime)

        # compute updated global attribute
        u_prime = self.f_u(e_agg, v_agg, u)

        return (u_prime, V_prime, E_prime)
