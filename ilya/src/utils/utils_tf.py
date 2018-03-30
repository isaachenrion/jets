import scipy as sp
import numpy as np
import utils.graph as graph
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops


def sp_sparse_to_tf_sparse(L):
    """
    Converts a scipy matrix into a tf one.
    """
    L = sp.sparse.csr_matrix(L)
    L = graph.rescale_L(L, lmax=2)
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    return tf.sparse_reorder(L)


@ops.RegisterGradient("FastSparseTensorDenseMatMul")
def _SparseTensorDenseMatMulGrad(op, grad):
    """Gradients for the dense tensor in the SparseTensorDenseMatMul op.
    If either input is complex, no gradient is provided.
    Args:
      op: the SparseTensorDenseMatMul op
      grad: the incoming gradient
    Returns:
      Gradient for each of the 4 input tensors:
        (sparse_indices, sparse_values, sparse_shape, dense_tensor)
      The gradients for indices and shape are None.
    Raises:
      TypeError: When the two operands don't have the same type.
    """
    sp_t = tf.SparseTensor(*op.inputs[:3])
    adj_a = op.get_attr("adjoint_a")
    adj_b = op.get_attr("adjoint_b")

    a_type = sp_t.values.dtype.base_dtype
    b_type = op.inputs[3].dtype.base_dtype
    if a_type != b_type:
        raise TypeError("SparseTensorDenseMatMul op received operands with "
                        "different types: ", a_type, " and ", b_type)

    # gradient w.r.t. dense
    b_grad = tf.sparse_tensor_dense_matmul(sp_t, grad,
                                                   adjoint_a=not adj_a)
    if adj_b:
        b_grad = tf.transpose(b_grad)

    # gradients w.r.t. (a_indices, a_values, a_shape, b)
    return (None, None, None, b_grad)


def chebyshev(inputs, L, k=5):
    batch_size, num_nodes, num_inputs = inputs.get_shape().as_list()
    assert k > 1

    # We need to multiply a Laplacian matrix L [N, N] with inputs
    # of shape [batch_size, N, num_inputs]
    # Thus, first we need to transform in into [N, batch_size * num_inputs]
    # and then restore to [batch_size, ]

    def my_map_fn(fn, arrays, dtype=tf.float32):
        # assumes all arrays have same leading dim
        indices = tf.range(tf.shape(arrays)[0])
        out = tf.map_fn(lambda ii: fn(arrays[ii]), indices, dtype=dtype)
        return out

    def my_matmul(L, x):
        def my_matmul_fn(a):
            g = tf.get_default_graph()
            with g.gradient_override_map({"SparseTensorDenseMatMul": "FastSparseTensorDenseMatMul"}):
                return tf.sparse_tensor_dense_matmul(L, a)

        return my_map_fn(my_matmul_fn, x)

    xs = [inputs]
    xs.append(my_matmul(L, xs[0]))

    for i in range(2, k):
        new_x = 2 * my_matmul(L, xs[-1]) - xs[-2]
        xs.append(new_x)

    # print(tf.concat_v2(xs, axis=2).get_shape())
    return tf.concat(xs, 2)


def chebyshev_(inputs, L, k=5):
    batch_size, num_nodes, num_inputs = inputs.get_shape().as_list()
    assert k > 1

    # We need to multiply a Laplacian matrix L [N, N] with inputs
    # of shape [batch_size, N, num_inputs]
    # Thus, first we need to transform in into [N, batch_size * num_inputs]
    # and then restore to [batch_size, ]

    # to [N, batch_size * num_inputs]
    x = tf.transpose(inputs, perm=[1, 0, 2])  # => [N, batch_size, num_inputs]
    x = tf.reshape(x, [num_nodes, batch_size * num_inputs])

    xs = [x]
    xs.append(tf.matmul(L, xs[0]))

    for i in range(2, k):
        new_x = 2 * tf.matmul(L, xs[-1]) - xs[-2]
        xs.append(new_x)

    for i in range(k):
        xs[i] = tf.reshape(xs[i], [num_nodes, batch_size, num_inputs])
        # => [batch_size, num_nodes, num_inputs]
        xs[i] = tf.transpose(xs[i], perm=[1, 0, 2])

    return tf.concat(xs, axis=2)


def conv_graph(inputs, L, num_outputs, k=5, activation_fn=tf.nn.relu):
    batch_size, num_nodes, num_inputs = inputs.get_shape().as_list()
    net = chebyshev(inputs, L=L, k=k)
    net = tf.reshape(net, [batch_size * num_nodes, -1])
    net = slim.fully_connected(net, num_outputs)
    net = activation_fn(net)
    net = tf.reshape(net, [batch_size, num_nodes, -1])
    return net


def max_pool_graph(inputs, p):
    return tf.nn.pool(inputs, [p], 'MAX', 'SAME', strides=[p])
