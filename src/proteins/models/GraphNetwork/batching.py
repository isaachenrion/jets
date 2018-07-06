import torch

def batch_graphs(u_list, V_list, E_list):
    graph_orders = [V.shape[0] for V in V_list]
    largest_graph_order = max(graph_orders)
    node_mask = torch.zeros(len(u_list) * largest_graph_order)
    for i, order in enumerate(graph_orders):
        offset = i*largest_graph_order
        node_mask[offset : offset + order] = 1

    new_e = []
    new_r = []
    new_s = []
    graph_sizes = []
    for i, E in enumerate(E_list):
        e, r, s = E
        graph_sizes.append(len(r))
        new_e.append(e)
        new_r.append(r + largest_graph_order * i)
        new_s.append(s + largest_graph_order * i)
    new_e = torch.cat(new_e, 0)
    new_r = torch.cat(new_r, 0)
    new_s = torch.cat(new_s, 0)
    new_E = new_e, new_r, new_s

    new_V = torch.zeros(largest_graph_order * len(V_list), *V_list[0].shape[1:])
    u_by_nodes = torch.zeros(largest_graph_order * len(V_list), *u_list[0].shape)
    for i, (V, u) in enumerate(zip(V_list, u_list)):
        offset = i*largest_graph_order
        new_V[offset:offset+len(V)] = V
        #print(u.shape)
        #print(len(V))
        u_by_nodes[offset:offset+len(V)] = u.expand(len(V), -1)

    new_u = torch.stack(u_list, 0)
    #u_by_nodes = torch.cat([u.expand(n,u.shape[1]) for n, u in zip(graph_orders, u_list)],0)
    u_by_edges = torch.cat([u.expand(m,u.shape[0]) for m, u in zip(graph_sizes, u_list)],0)

    edge_offsets = torch.cumsum(torch.tensor([0]+graph_sizes), 0)
    return new_u, new_V, new_E, torch.tensor(graph_orders), node_mask, edge_offsets, u_by_nodes, u_by_edges
