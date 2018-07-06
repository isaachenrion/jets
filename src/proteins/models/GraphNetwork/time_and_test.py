import numpy as np
from .generate_graph_data import generate_graph_data
from .GraphNetworkBlockUnbatched import GraphNetworkBlockUnbatched
from .GraphNetworkBlock import GraphNetworkBlock

def time_gn_batched_versus_unbatched(batch_size, n_trials, max_data_dim):
    t_u = 0
    t_b = 0
    for i in range(n_trials):
        graph_dim = np.random.randint(2, max_data_dim)
        edge_dim = np.random.randint(2, max_data_dim)
        node_dim = np.random.randint(2, max_data_dim)
        n_nodes = np.random.randint(2, max_data_dim)
        n_layers = np.random.randint(2, max_data_dim)
        hidden_dim = np.random.randint(2, max_data_dim)

        u_list = []; V_list = []; E_list = []
        for i in range(batch_size):
            u, V, E = generate_graph_data(graph_dim, node_dim, edge_dim, n_nodes + i)
            u_list.append(u)
            V_list.append(V)
            E_list.append(E)

        gnbu = GraphNetworkBlockUnbatched(graph_dim, node_dim, edge_dim)

        t = time.time()
        unbatched_out_u_list = []; unbatched_out_E_list = []; unbatched_out_V_list = []
        for _u, _V, _E in zip(u_list, V_list, E_list):
            u_, V_, E_ = gnbu(_u,_V,_E)
            unbatched_out_u_list.append(u_)
            unbatched_out_V_list.append(V_)
            unbatched_out_E_list.append(E_)
        t_u = t_u + time.time() - t

        gnbb = GraphNetworkBlock(graph_dim, node_dim, edge_dim)
        gnbb.load_state_dict(gnbu.state_dict())

        t = time.time()
        u_b, V_b, E_b, go_b, nm_b, eo_b, un_b, ue_b = batch_graphs(u_list, V_list, E_list)
        b_u_b, b_V_b, b_E_b = gnbb(u_b, V_b, E_b, go_b, nm_b, eo_b, un_b, ue_b)
        t_b = t_b + time.time() - t

    t_u /= n_trials
    t_b /= n_trials

    print("Did {} runs of each.".format(n_trials))
    print("Unbatched took {:.3f}s on average".format(t_u))
    print("Batched took {:.3f}s on average".format(t_b))

def test_gn_batched_equals_unbatched(batch_size=17, n_trials=10, max_data_dim=97):
    for i in range(n_trials):
        graph_dim = np.random.randint(2, max_data_dim)
        edge_dim = np.random.randint(2, max_data_dim)
        node_dim = np.random.randint(2, max_data_dim)
        n_nodes = np.random.randint(2, max_data_dim)
        n_layers = np.random.randint(2, max_data_dim)
        hidden_dim = np.random.randint(2, max_data_dim)

        u_list = []; V_list = []; E_list = []
        for i in range(batch_size):
            u, V, E = generate_graph_data(graph_dim, node_dim, edge_dim, n_nodes + i)
            u_list.append(u)
            V_list.append(V)
            E_list.append(E)
        print("Running unbatched forward")
        gnbu = GraphNetworkBlockUnbatched(graph_dim, node_dim, edge_dim)
        unbatched_out_u_list = []; unbatched_out_E_list = []; unbatched_out_V_list = []
        for _u, _V, _E in zip(u_list, V_list, E_list):
            u_, V_, E_ = gnbu(_u,_V,_E)
            unbatched_out_u_list.append(u_)
            unbatched_out_V_list.append(V_)
            unbatched_out_E_list.append(E_)
        ub_u_b, ub_V_b, ub_E_b, ub_go_b, ub_nm_b, ub_eo_b, ub_un_b, ub_ue_b = batch_graphs(unbatched_out_u_list, unbatched_out_V_list, unbatched_out_E_list)

        gnbb = GraphNetworkBlock(graph_dim, node_dim, edge_dim)
        gnbb.load_state_dict(gnbu.state_dict())

        u_b, V_b, E_b, go_b, nm_b, eo_b, un_b, ue_b = batch_graphs(u_list, V_list, E_list)

        print("Running batched forward")
        b_u_b, b_V_b, b_E_b = gnbb(u_b, V_b, E_b, go_b, nm_b, eo_b, un_b, ue_b)

        fail = False
        try:
            assert (b_u_b == ub_u_b).all()
        except AssertionError:
            print("u fail")
            print((b_u_b - ub_u_b).abs().mean())
            fail = True
        try:
            assert (b_V_b == ub_V_b).all()
        except AssertionError:
            print("V fail")
            print((b_V_b - ub_V_b).abs().mean())
            fail = True
        try:
            assert (b_E_b[0] == ub_E_b[0]).all()
        except AssertionError:
            print("E fail")
            print((b_E_b[0] - ub_E_b[0]).abs().mean())
            fail = True

        if not fail:
            print("All checks passed!")
