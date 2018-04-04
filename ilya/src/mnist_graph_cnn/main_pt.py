import sys
sys.path.append('..')
import utils.graph as graph
import utils.coarsening as coarsening
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy as sp
import time
import utils.utils_pt as utils
from progressbar import ProgressBar
from torch.autograd import Variable
from functools import reduce

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--model', default="cheb",
                    help='cheb | op')
parser.add_argument('--num-epoch', type=int, default=100, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--number-edges', type=int, default=8, metavar='N',
                    help='minimum number of edges per vertex (default: 8)')
parser.add_argument('--coarsening-levels', type=int, default=4, metavar='N',
                    help='number of coarsened graphs. (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(42)
if args.cuda:
    torch.cuda.manual_seed(42)

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
W = graphs
D = []

for i in range(len(L)):
    L[i] = graph.rescale_L(L[i], lmax=2)

    L[i] = utils.sp_sparse_to_pt_sparse(L[i])
    L[i] = utils.to_dense_batched(L[i], args.batch_size)

    D.append(sp.sparse.diags(W[i].sum(0).A.squeeze(), 0))
    D[i] = utils.sp_sparse_to_pt_sparse(D[i])
    D[i] = utils.to_dense_batched(D[i], args.batch_size)

    W[i] = utils.sp_sparse_to_pt_sparse(W[i])
    W[i] = utils.to_dense_batched(W[i], args.batch_size)

    if args.cuda:
        L[i] = L[i].cuda()
        D[i] = D[i].cuda()
        W[i] = W[i].cuda()

mnist = input_data.read_data_sets('../data', one_hot=False)

train_data = mnist.train.images.astype(np.float32)
val_data = mnist.validation.images.astype(np.float32)
test_data = mnist.test.images.astype(np.float32)
train_labels = torch.from_numpy(mnist.train.labels)
val_labels = torch.from_numpy(mnist.validation.labels)
test_labels = torch.from_numpy(mnist.test.labels)

t_start = time.process_time()
train_data = torch.from_numpy(coarsening.perm_data(train_data, perm))
val_data = torch.from_numpy(coarsening.perm_data(val_data, perm))
test_data = torch.from_numpy(coarsening.perm_data(test_data, perm))
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
del perm

# Ilya:
# All code above including libraries is taken from https://github.com/mdeff/cnn_graph
# Data is preprocessed in such a way that we can perform a simple pooling
# operation in order to coarsen it

class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()

        if args.model == "cheb":
            self.graph_conv1 = utils.GraphConv(1, 32, k=25)
            self.bn1 = utils.GraphBatchNorm(32)
            self.graph_conv2 = utils.GraphConv(32, 64, k=25)
            self.bn2 = utils.GraphBatchNorm(64)
        else:
            self.graph_conv1 = utils.LapConv(1, 32)
            self.bn1 = utils.GraphBatchNorm(32)
            self.graph_conv2 = utils.LapConv(32, 32)
            self.bn2 = utils.GraphBatchNorm(32)
            self.graph_conv3 = utils.LapConv(32, 32)
            self.bn3 = utils.GraphBatchNorm(32)
            self.graph_conv4 = utils.LapConv(32, 32)
            self.bn4 = utils.GraphBatchNorm(32)
            self.graph_conv5 = utils.LapConv(32, 32)
            self.bn5 = utils.GraphBatchNorm(32)

            self.graph_conv6 = utils.LapConv(32, 64)
            self.bn6 = utils.GraphBatchNorm(64)
            self.graph_conv7 = utils.LapConv(64, 64)
            self.bn7 = utils.GraphBatchNorm(64)
            self.graph_conv8 = utils.LapConv(64, 64)
            self.bn8 = utils.GraphBatchNorm(64)
            self.graph_conv9 = utils.LapConv(64, 64)
            self.bn9 = utils.GraphBatchNorm(64)
            self.graph_conv10 = utils.LapConv(64, 64)
            self.bn10 = utils.GraphBatchNorm(64)

        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc1.bias.data.zero_()

        self.do = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.fc2.bias.data.zero_()

    def forward(self, x, *Ls):
        x = x.unsqueeze(2)

        batch_size, num_nodes, num_inputs = x.size()

        if args.model == "cheb":
            x = self.graph_conv1(Ls[0], x)
            x = self.bn1(x)
            x = F.relu(x)
        else:
            x = self.graph_conv1(Ls[0], x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.graph_conv2(Ls[0], x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.graph_conv3(Ls[0], x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.graph_conv4(Ls[0], x)
            x = self.bn4(x)
            x = F.relu(x)
            x = self.graph_conv5(Ls[0], x)
            x = self.bn5(x)
            x = F.relu(x)

        x = x.unsqueeze(1)
        x = F.max_pool2d(x, [4, 1])
        x = x.squeeze()

        if args.model == "cheb":
            x = self.graph_conv2(Ls[2], x)
            x = self.bn2(x)
            x = F.relu(x)
        else:
            x = self.graph_conv6(Ls[2], x)
            x = self.bn6(x)
            x = F.relu(x)
            x = self.graph_conv7(Ls[2], x)
            x = self.bn7(x)
            x = F.relu(x)
            x = self.graph_conv8(Ls[2], x)
            x = self.bn8(x)
            x = F.relu(x)
            x = self.graph_conv9(Ls[2], x)
            x = self.bn9(x)
            x = F.relu(x)
            x = self.graph_conv10(Ls[2], x)
            x = self.bn10(x)
            x = F.relu(x)

        x = x.unsqueeze(1)
        x = F.max_pool2d(x, [4, 1])
        x = x.squeeze()

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.do(x)

        x = self.fc2(x)
        x = F.log_softmax(x)

        return x

model = Model(64 * L[4].size(1))
#model = torch.nn.DataParallel(model)
torch.set_num_threads(12)

print("Params #: {}".format(reduce((lambda x, y: x + y), [param.view(-1).size(0) for param in model.parameters()])))

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.9)

inputs_np = torch.zeros([args.batch_size, train_data.size(1)])
labels_np = torch.zeros([args.batch_size]).long()

if args.cuda:
    inputs_np, labels_np = inputs_np.cuda(), labels_np.cuda()

lr = 1e-2
for e in range(args.num_epoch):
    model.train()
    pbar = ProgressBar()

    num_correct = 0
    num_overall = 0
    for i in pbar(range(train_data.size(0) // args.batch_size)):
        for b in range(args.batch_size):
            ind = np.random.randint(train_data.size(0))
            inputs_np[b] = train_data[ind]
            labels_np[b] = train_labels[ind]

        output = model(Variable(inputs_np), *[Variable(l) for l in L])

        optimizer.zero_grad()
        loss = F.nll_loss(output, Variable(labels_np))
        loss.backward()
        optimizer.step()

        pred = output.data.max(1)[1] # get the index of the max log-probability
        num_correct += pred.eq(labels_np).cpu().sum()
        num_overall += args.batch_size

    lr *= 0.95
    print("Epoch {}: train accuracy {}, learning rate {:.5f}".format(e, num_correct / num_overall * 100, lr))

    model.eval()
    pbar = ProgressBar()

    num_correct = 0
    num_overall = 0
    for i in pbar(range(test_data.size(0) // args.batch_size)):
        inputs_np[:] = test_data[i * args.batch_size: (i + 1) * args.batch_size]
        labels_np[:] = test_labels[i * args.batch_size: (i + 1) * args.batch_size]

        output = model(Variable(inputs_np, volatile=True), *[Variable(l, volatile=True) for l in L])

        pred = output.data.max(1)[1] # get the index of the max log-probability
        num_correct += pred.eq(labels_np).cpu().sum()
        num_overall += args.batch_size

    print("Test accuracy {}".format(num_correct / num_overall * 100))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
