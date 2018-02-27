from .message_passing_layers import construct_mp_layer
#from .adjacency import construct_adjacency_matrix_layer

'''
This module implements the core message passing operations.

###adjacency.py <-- compute an adjacency matrix based on vertex data.
message_passing.py <-- run a single iteration of message passing.
message.py <-- compute a message, given a hidden state.
vertex_update.py <-- compute a vertex's new hidden state, given a message.
'''
