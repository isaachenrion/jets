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

        self.f_e_nn = nn.Linear(edge_dim + 2 * node_dim + graph_dim, edge_dim_out)
        self.f_v_nn = nn.Linear(edge_dim_out + node_dim + graph_dim, node_dim_out)
        self.f_u_nn = nn.Linear(edge_dim_out + node_dim_out + graph_dim, graph_dim_out)

    def rho_e_v(self, E_prime_i):
        return E_prime_i.mean(0)

    def rho_e_u(self, E_prime, edge_offsets):
        e_prime, s, r = E_prime
        e_agg = []
        for i, _ in enumerate(edge_offsets[:-1]):
            e_agg.append(e_prime[edge_offsets[i]:edge_offsets[i+1]].mean(0))
        e_agg = torch.stack(e_agg, 0)
        return e_agg

    def rho_v_u(self, V_prime, graph_orders):
        largest_graph_order = max(graph_orders)
        v_agg = V_prime.view(-1, largest_graph_order, *V_prime.shape[1:]).sum(1)
        v_agg = v_agg / graph_orders.float().unsqueeze(1)
        return v_agg

    def f_e(self, e, v_s, v_r, u):
        x = torch.cat([e,v_s,v_r,u], 1)
        x = self.f_e_nn(x)
        x = F.relu(x)
        return x

    def f_u(self, e_agg, v_agg, u):
        x = torch.cat([e_agg,v_agg,u], 1).unsqueeze(0)
        x = self.f_u_nn(x)
        x = F.relu(x)
        return x

    def f_v(self, e_agg_by_node, v, u, node_mask):
        x = torch.cat([e_agg_by_node,v,u], 1)
        x = self.f_v_nn(x)
        x = F.relu(x)
        x = x * node_mask.unsqueeze(1)
        return x


    def forward(self, u, V, E, graph_orders, node_mask, edge_offsets, u_by_nodes, u_by_edges):
        e, s, r = E
        #v = V

        v_s = torch.index_select(V, 0, s)
        v_r = torch.index_select(V, 0, r)

        # compute updated edge attributes
        e_prime = self.f_e(e, v_s, v_r, u_by_edges)

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
        V_prime = self.f_v(e_agg_by_node, V, u_by_nodes, node_mask)

        E_prime = e_prime, s, r

        # aggregate edge attributes globally
        e_agg = self.rho_e_u(E_prime, edge_offsets)

        # aggregate node attributes globally
        v_agg = self.rho_v_u(V_prime, graph_orders)

        # compute updated global attribute
        u_prime = self.f_u(e_agg, v_agg, u)

        return (u_prime, V_prime, E_prime)
