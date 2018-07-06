
from .batching import batch_graphs

# batching checks
def test_batching_works(batch_size=17, n_trials=10, max_data_dim=97):
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

        u_b, V_b, E_b, go_b, nm_b, eo_b, un_b, ue_b = batch_graphs(u_list, V_list, E_list)

        try:
            for i, (_u_u, _V_u, _E_u) in enumerate(zip(u_list, V_list, E_list)):
                _u_b, _V_b, _E_b = u_b[i], V_b[i*go_b:(i+1)*go_b], E_b[0][eo_b[i]:eo_b[i+1]]
                assert (_u_b == _u_u).all()
                assert (_V_u == _V_b[:_V_u.shape[0]]).all()
                assert (_E_u[0] == _E_b[:_E_u[0].shape[0]]).all()
        except AssertionError:
            print("FAIL")
