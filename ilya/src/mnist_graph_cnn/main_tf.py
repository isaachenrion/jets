import sys
sys.path.append('..')
import utils.graph as graph
import utils.coarsening as coarsening
import argparse
import tensorflow as tf
from tensorflow.contrib import slim
import torch
import torch.utils.data
from torchvision import datasets, transforms
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
import utils.utils_tf as utils
from progressbar import ProgressBar

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-epoch', type=int, default=100, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--number-edges', type=int, default=8, metavar='N',
                    help='minimum number of edges per vertex (default: 8)')
parser.add_argument('--coarsening-levels', type=int, default=4, metavar='N',
                    help='number of coarsened graphs. (default: 4)')
args = parser.parse_args()

# Preprocessing for mnist

def grid_graph(m, corners=False):
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=args.number_edges, metric='euclidean')
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz//2, args.number_edges*m**2//2))
    return A

A = grid_graph(28, corners=False)
A = graph.replace_random_edges(A, 0)
graphs, perm = coarsening.coarsen(A, levels=args.coarsening_levels, self_connections=False)
L = [graph.laplacian(A, normalized=True) for A in graphs]
del A

mnist = input_data.read_data_sets('../data', one_hot=False)

train_data = mnist.train.images.astype(np.float32)
val_data = mnist.validation.images.astype(np.float32)
test_data = mnist.test.images.astype(np.float32)
train_labels = mnist.train.labels
val_labels = mnist.validation.labels
test_labels = mnist.test.labels

t_start = time.process_time()
train_data = coarsening.perm_data(train_data, perm)
val_data = coarsening.perm_data(val_data, perm)
test_data = coarsening.perm_data(test_data, perm)
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
del perm

# Ilya:
# All code above including libraries is taken from https://github.com/mdeff/cnn_graph
# Data is preprocessed in such a way that we can perform a simple pooling
# operation in order to coarsen it

# My code starts below

inputs_tf = tf.placeholder(tf.float32, [args.batch_size, train_data.shape[1]]) # batch size x nodes
labels_tf = tf.placeholder(tf.int64, [args.batch_size])
L_tf = []

for i in range(0, len(L)):
    L_tf.append(utils.sp_sparse_to_tf_sparse(L[i]))
    #L_tf[-1] = tf.sparse_tensor_to_dense(L_tf[-1])
    #L_tf[-1] = tf.expand_dims(L_tf[-1], 0)
    #L_tf[-1] = tf.tile(L_tf[-1], [args.batch_size, 1, 1])

def get_model(inputs, is_training=True):
    # Define the model
    net = tf.expand_dims(inputs, 2)
    net = utils.conv_graph(net, L_tf[0], 32, k=25)
    net = utils.max_pool_graph(net, 4)

    net = utils.conv_graph(net, L_tf[2], 64, k=25)
    net = utils.max_pool_graph(net, 4)

    net = slim.flatten(net)

    net = slim.fully_connected(net, 512)

    net = slim.dropout(net, is_training=is_training)

    return slim.fully_connected(net, 10, activation_fn=None)

with tf.variable_scope("model"):
    train_logits = get_model(inputs_tf)

with tf.variable_scope("model", reuse=True):
    test_logits = get_model(inputs_tf, is_training=False)

loss = slim.losses.sparse_softmax_cross_entropy(train_logits, labels_tf)

predictions = tf.arg_max(test_logits, 1)
accuracy = tf.contrib.metrics.accuracy(predictions, labels_tf)

lr_tf = tf.placeholder(tf.float32, ())
optimizer = tf.train.MomentumOptimizer(lr_tf, momentum=0.9) # Default params

train_op = slim.learning.create_train_op(loss, optimizer)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

init = tf.global_variables_initializer()
sess.run(init)

inputs_np = np.zeros([args.batch_size, train_data.shape[1]])
labels_np = np.zeros([args.batch_size], dtype='int32')

lr = 3e-2
for e in range(args.num_epoch):
    pbar = ProgressBar()

    num_correct = 0
    num_overall = 0
    for i in pbar(range(train_data.shape[0] // args.batch_size)):
        for b in range(args.batch_size):
            ind = np.random.randint(train_data.shape[0])
            inputs_np[b] = train_data[ind]
            labels_np[b] = train_labels[ind]

        _, accuracy_batch = sess.run([train_op, accuracy], {inputs_tf: inputs_np, labels_tf: labels_np, lr_tf: lr})
        num_correct += accuracy_batch * args.batch_size
        num_overall += args.batch_size

    lr *= 0.95
    print("Epoch {}: train accuracy {}, learning rate {:.5f}".format(e, num_correct / num_overall * 100, lr))

    pbar = ProgressBar()

    num_correct = 0
    num_overall = 0
    for i in pbar(range(test_data.shape[0] // args.batch_size)):
        inputs_np[:] = test_data[i * args.batch_size: (i + 1) * args.batch_size]
        labels_np[:] = test_labels[i * args.batch_size: (i + 1) * args.batch_size]

        accuracy_batch = sess.run(accuracy, {inputs_tf: inputs_np, labels_tf: labels_np})
        num_correct += accuracy_batch * args.batch_size
        num_overall += args.batch_size

    print("Test accuracy {}".format(num_correct / num_overall * 100))
