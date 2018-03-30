import os, sys
sys.path.append(os.path.dirname(__file__))
#from cuda.sparse_bmm_func import SparseBMMFunc

import scipy as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sparse_cat(tensors, size0, size1):
    '''
    Input: list of sparse matrices
    size0 = # rows
    size1 = # cols
    Output: 3d tensor (BS, R, C)
    '''
    values = []
    tensor_size = 0
    for i, tensor in enumerate(tensors):
        values.append(tensor._values())
        tensor_size += tensor._nnz()

    indices = torch.LongTensor(3, tensor_size)
    index = 0
    for i, tensor in enumerate(tensors):
        tensor = tensor.coalesce()
        indices[0, index:index+tensor._nnz()] = i
        indices[1:3, index:index+tensor._nnz()].copy_(tensor._indices())

        index += tensor._nnz()

    values = torch.cat(values, 0)

    size = torch.Size((len(tensors), size0, size1))
    return torch.sparse.FloatTensor(indices, values, size)


def sp_sparse_to_pt_sparse(L):
    """
    Converts a scipy matrix into a pt one.
    """
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    indices = torch.from_numpy(indices).long()
    L_data = torch.from_numpy(L.data)

    size = torch.Size(L.shape)
    indices = indices.transpose(1, 0)

    L = torch.sparse.FloatTensor(indices, L_data, size)
    return L

def to_dense_batched(x, batch_size):
    x = x.to_dense()
    x = x.unsqueeze(0)
    return x.repeat(batch_size, 1, 1)

import time


class Chebyshev(nn.Module):
    def __init__(self, k):
        super(Chebyshev, self).__init__()
        self.k = k

    def forward(self, L, x):
        batch_size, num_nodes, num_inputs = x.size()

        """
        # to [N, batch_size * num_inputs]
        x = tf.transpose(inputs, perm=[1, 0, 2]) # => [N, batch_size, num_inputs]
        x = tf.reshape(x, [num_nodes, batch_size * num_inputs])

        xs = [x]
        xs.append(tf.matmul(L, xs[0]))

        for i in range(k):
            xs[i] = tf.reshape(xs[i], [num_nodes, batch_size, num_inputs])
            xs[i] = tf.transpose(xs[i], perm=[1, 0, 2]) # => [batch_size, num_nodes, num_inputs]

        """
        #x = x.transpose(1, 0).contiguous()
        #x = x.view(num_nodes, batch_size * num_inputs)

        #L = L.cpu()
        xs = [x]
        if self.k > 1:
            xs.append(torch.bmm(L, xs[0]))

        for i in range(2, self.k):
            new_x = 2 * torch.bmm(L, xs[-1]) - xs[-2]
            xs.append(new_x)

        """
        for i in range(self.k):
            xs[i] = xs[i].view(num_nodes, batch_size, num_inputs)
            xs[i] = xs[i].transpose(1, 0) # => [batch_size, num_nodes, num_inputs]
        """
        outputs = torch.cat(xs, 2)
        return outputs

class Chebyshev_(nn.Module):
    def __init__(self, k):
        super(Chebyshev, self).__init__()
        self.k = k

    def forward(self, L, x):
        batch_size, num_nodes, num_inputs = x.size()

        xs = [x]
        xs.append(torch.bmm(L, xs[0]))

        for i in range(2, self.k):
            new_x = 2 * torch.bmm(L, xs[-1]) - xs[-2]
            xs.append(new_x)

        return torch.cat(xs, 2)

class GraphConv(nn.Module):
    def __init__(self, num_inputs, num_outputs, k=25):
        super(GraphConv, self).__init__()
        self.conv = Chebyshev(k)
        self.conv_fc = nn.Linear(k * num_inputs, num_outputs)
        self.conv_fc.bias.data.zero_()

        self.k = k
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def forward(self, L, x):
        x = self.conv(L, x)
        batch_size, num_nodes, num_inputs = x.size()
        x = x.view(batch_size * num_nodes, self.k * self.num_inputs)
        x = self.conv_fc(x)
        x = x.view(batch_size, num_nodes, self.num_outputs)
        return x

class GraphConv1x1(nn.Module):
    def __init__(self, num_inputs, num_outputs, batch_norm=None):
        super(GraphConv1x1, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch_norm = batch_norm

        if self.batch_norm == "pre":
            self.bn = nn.BatchNorm1d(num_inputs)

        if self.batch_norm == "post":
            self.bn = nn.BatchNorm1d(num_outputs)

        self.fc = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        batch_size, num_nodes, num_inputs = x.size()
        assert num_inputs == self.num_inputs

        x = x.contiguous()
        x = x.view(-1, self.num_inputs)

        if self.batch_norm == "pre":
            x = self.bn(x)
        x = self.fc(x)
        if self.batch_norm == "post":
            x = self.bn(x)

        x = x.view(batch_size, num_nodes, self.num_outputs)
        return x


"""
def chebyshev(inputs, L, k=5):
    batch_size, num_nodes, num_inputs = inputs.get_shape().as_list()
    assert k > 1

    # We need to multiply a Laplacian matrix L [N, N] with inputs
    # of shape [batch_size, N, num_inputs]
    # Thus, first we need to transform in into [N, batch_size * num_inputs]
    # and then restore to [batch_size, ]


    # to [N, batch_size * num_inputs]
    x = tf.transpose(inputs, perm=[1, 0, 2]) # => [N, batch_size, num_inputs]
    x = tf.reshape(x, [num_nodes, batch_size * num_inputs])

    xs = [x]
    xs.append(tf.matmul(L, xs[0]))

    for i in range(2, k):
        new_x = 2 * tf.matmul(L, xs[-1]) - xs[-2]
        xs.append(new_x)

    for i in range(k):
        xs[i] = tf.reshape(xs[i], [num_nodes, batch_size, num_inputs])
        xs[i] = tf.transpose(xs[i], perm=[1, 0, 2]) # => [batch_size, num_nodes, num_inputs]

    return tf.concat_v2(xs, axis=2)
"""

class GraphBatchNorm(nn.Module):
    def __init__(self, num_inputs):
        super(GraphBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_inputs)

    def forward(self, x):
        self.bn.train()
        batch_size, num_nodes, num_inputs = x.size()
        x = x.view(batch_size * num_nodes, num_inputs)
        x = self.bn(x)
        x = x.view(batch_size, num_nodes, num_inputs)
        return x


class GraphResNet(nn.Module):
    def __init__(self, num_outputs, conv_module, num_operators):
        super(GraphResNet, self).__init__()
        raise Exception("Depricated")
        self.num_outputs = num_outputs

        self.bn0 = GraphBatchNorm(num_operators * num_outputs)

        self.conv1 = conv_module(num_outputs, num_outputs)
        self.bn1 = GraphBatchNorm(num_operators * num_outputs)

        self.conv2 = conv_module(num_outputs, num_outputs)
        self.num_operators = num_operators

    def forward(self, L, mask, inputs):
        x = torch.cat(inputs, 2)
        x = self.bn0(x)
        x = x.split(self.num_outputs, 2)

        x_sum = 0
        for i in range(self.num_operators):
            x_sum = x_sum + x[i]
        x = F.elu(x_sum)

        x = self.conv1(L, mask, x)
        x = torch.cat(x, 2)
        x = self.bn1(x)
        x = x.split(self.num_outputs, 2)

        x_sum = 0
        for i in range(self.num_operators):
            x_sum = x_sum + x[i]
        x = F.elu(x_sum)

        x = self.conv2(L, mask, x)

        outputs = []
        for i in range(self.num_operators):
            outputs.append(inputs[i] + x[i])
        return outputs

class LapConv(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LapConv, self).__init__()
        raise Exception("Depricated")

        self.conv_fc1 = nn.Linear(num_inputs, num_outputs)
        self.conv_fc2 = nn.Linear(num_inputs, num_outputs)
        self.conv_fc3 = nn.Linear(num_inputs, num_outputs)

        self.conv_fcs = [self.conv_fc1, self.conv_fc2, self.conv_fc3]

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def forward(self, L, mask, x):
        if L.data.is_sparse:
            xs = [x, SparseBMMFunc()(L, x)]
        else:
            xs = [x, torch.bmm(L, x)]

        xs += [global_average(x, mask).expand_as(x).contiguous()]

        results = []
        for i, x in enumerate(xs):
            batch_size, num_nodes, _ = x.size()

            x = x.view(batch_size * num_nodes, -1)
            x = self.conv_fcs[i](x)
            x = x.view(batch_size, num_nodes, self.num_outputs)
            results.append(x)
        return results


def global_average(x, mask):
    mask = mask.expand_as(x)
    return (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)

class AvgConv(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(AvgConv, self).__init__()
        raise Exception("Depricated")

        self.conv_fc1 = nn.Linear(num_inputs, num_outputs)
        self.conv_fc2 = nn.Linear(num_inputs, num_outputs)

        self.conv_fcs = [self.conv_fc1, self.conv_fc2]

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def forward(self, L, mask, x):
        xs = [x, global_average(x, mask).expand_as(x).expand_as(x).contiguous()]

        results = []
        for i, x in enumerate(xs):
            batch_size, num_nodes, _ = x.size()

            x = x.contiguous().view(batch_size * num_nodes, -1)
            x = self.conv_fcs[i](x)
            x = x.view(batch_size, num_nodes, self.num_outputs)
            results.append(x)
        return results

class DirConv(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DirConv, self).__init__()
        raise Exception("Depricated")

        self.fc1 = nn.Linear(num_inputs, num_outputs)
        self.fc2 = nn.Linear(num_inputs, num_outputs)
        self.fc3 = nn.Linear(num_inputs, num_outputs)
        self.fc4 = nn.Linear(num_inputs, num_outputs)
        self.fc5 = nn.Linear(num_inputs, num_outputs)

        self.fcs = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def forward(self, L, Di, DiA, v, f):
        batch_size, num_nodes, num_inputs = v.size()
        _, num_faces, _ = f.size()

        v_ = v.view(batch_size, num_nodes * 4, num_inputs // 4)
        xs = []
        if Di.data.is_sparse:
            xs += [f, SparseBMMFunc()(Di, v_).view(batch_size, num_faces, num_inputs)]
        else:
            xs += [f, torch.bmm(Di, v_).view(batch_size, num_faces, num_inputs)]

        f = f.view(batch_size, num_faces * 4, num_inputs // 4)
        if DiA.data.is_sparse:
            xs += [v, SparseBMMFunc()(DiA, f).view(batch_size, num_nodes, num_inputs), SparseBMMFunc()(L, v)]
        else:
            xs += [v, torch.bmm(DiA, f).view(batch_size, num_nodes, num_inputs), torch.bmm(L, v)]

        result = []
        for i, x in enumerate(xs):
            x = x.view(-1, self.num_inputs)
            x = self.fcs[i](x)
            x = x.view(batch_size, -1, self.num_outputs)
            result.append(x)

        return result

class DirResNet(nn.Module):
    def __init__(self, num_outputs):
        super(DirResNet, self).__init__()
        self.num_outputs = num_outputs
        raise Exception("Depricated")

        self.bn0_f = GraphBatchNorm(2 * num_outputs)
        self.bn0_v = GraphBatchNorm(3 * num_outputs)

        self.conv1 = DirConv(num_outputs, num_outputs)

        self.bn1_f = GraphBatchNorm(2 * num_outputs)
        self.bn1_v = GraphBatchNorm(3 * num_outputs)

        self.conv2 = DirConv(num_outputs, num_outputs)

    def forward(self, L, Di, DiA, inputs):
        f = torch.cat([inputs[0], inputs[1]], 2)
        v = torch.cat([inputs[2], inputs[3], inputs[4]], 2)

        f = self.bn0_f(f)
        v = self.bn0_v(v)
        f = f.split(self.num_outputs, 2)
        v = v.split(self.num_outputs, 2)
        assert len(f) == 2
        assert len(v) == 3
        v, f = F.elu(v[0] + v[1] + v[2]), F.elu(f[0] + f[1])

        x = self.conv1(L, Di, DiA, v, f)
        f = torch.cat([x[0], x[1]], 2)
        v = torch.cat([x[2], x[3], x[4]], 2)
        f = self.bn1_f(f)
        v = self.bn1_v(v)
        f = f.split(self.num_outputs, 2)
        v = v.split(self.num_outputs, 2)
        assert len(f) == 2
        assert len(v) == 3
        v, f = F.elu(v[0] + v[1] + v[2]), F.elu(f[0] + f[1])

        x = self.conv2(L, Di, DiA, v, f)
        return [x[0] + inputs[0], x[1] + inputs[1], x[2] + inputs[2], x[3] + inputs[3], x[4] + inputs[4]]


class LapResNet2(nn.Module):
    def __init__(self, num_outputs):
        super(LapResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")

    def forward(self, L, mask, inputs):
        x = inputs
        x = F.elu(x)

        xs = [x, SparseBMMFunc()(L, x)]#, global_average(x, mask).expand_as(x).contiguous()]

        x = torch.cat(xs, 2)
        x = self.bn_fc0(x)

        x = F.elu(x)
        xs = [x, SparseBMMFunc()(L, x)]#, global_average(x, mask).expand_as(x).contiguous()]
        x = torch.cat(xs, 2)
        x = self.bn_fc1(x)

        return x + inputs

class DirResNet2(nn.Module):
    def __init__(self, num_outputs, res_f=False):
        super(DirResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        #self.bn_fc2 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.res_f = res_f

    def forward(self, Di, DiA, v, f):
        batch_size, num_nodes, num_inputs = v.size()
        _, num_faces, _ = f.size()

        """
        v0 = F.elu(v)
        v0 = v0.view(batch_size, num_nodes * 4, num_inputs // 4)
        f0 = SparseBMMFunc()(Di, v0)#, global_average(x, mask).expand_as(x).contiguous()]
        f0 = f0.view(batch_size, num_faces, num_inputs)
        f0 = torch.cat([f, f0], 2)
        f0 = self.bn_fc_f0(f0)

        f0 = F.elu(f)
        f0_ = f0.view(batch_size, num_faces * 4, num_inputs // 4)
        v0 = SparseBMMFunc()(DiA, f0_)#, global_average(x, mask).expand_as(x).contiguous()]
        v0 = v0.view(batch_size, num_nodes, num_inputs)
        v0 = torch.cat([v, v0], 2)
        v0 = self.bn_fc_v0(v0)

        v1 = F.elu(v0)
        v1_ = v1.view(batch_size, num_nodes * 4, num_inputs // 4)
        f1 = SparseBMMFunc()(Di, v1_)#, global_average(x, mask).expand_as(x).contiguous()]
        f1 = f1.view(batch_size, num_faces, num_inputs)
        f1 = torch.cat([f0, f1], 2)
        f1 = self.bn_fc_f1(f1)

        f1 = F.elu(f0)
        f1 = f1.view(batch_size, num_faces * 4, num_inputs // 4)
        v1 = SparseBMMFunc()(DiA, f1)#, global_average(x, mask).expand_as(x).contiguous()]
        v1 = v1.view(batch_size, num_nodes, num_inputs)
        v1 = torch.cat([v0, v1], 2)
        v1 = self.bn_fc_v1(v1)
        return v + v0, f + f1
        """

        x_in, f_in = F.elu(v), F.elu(f)
        x = x_in
        x = x.view(batch_size, num_nodes * 4, num_inputs // 4)
        x = SparseBMMFunc()(Di, x)#, global_average(x, mask).expand_as(x).contiguous()]
        x = x.view(batch_size, num_faces, num_inputs)
        x = torch.cat([f_in, x], 2)
        x = self.bn_fc0(x)
        f_out = x

        x = F.elu(x)
        x = x.view(batch_size, num_faces * 4, num_inputs // 4)
        x = SparseBMMFunc()(DiA, x)#, global_average(x, mask).expand_as(x).contiguous()]
        x = x.view(batch_size, num_nodes, num_inputs)
        x = torch.cat([x_in, x], 2)
        x = self.bn_fc1(x)
        v_out = x
        """
        x = F.elu(x)
        x = x.view(batch_size, num_faces * 4, num_inputs // 4)
        x = SparseBMMFunc()(DiA, x)#, global_average(x, mask).expand_as(x).contiguous()]
        x = x.view(batch_size, num_nodes, num_inputs)
        x = torch.cat([f_int, x], 2)
        x = self.bn_fc2(x)
        """
        return v + v_out, f_out

class AvgResNet2(nn.Module):
    def __init__(self, num_outputs):
        super(AvgResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")

    def forward(self, L, mask, inputs):
        x = inputs
        x = F.elu(x)

        xs = [x, global_average(x, mask).expand_as(x).contiguous()]
        x = torch.cat(xs, 2)
        x = self.bn_fc0(x)

        x = F.elu(x)
        xs = [x, global_average(x, mask).expand_as(x).contiguous()]
        x = torch.cat(xs, 2)
        x = self.bn_fc1(x)

        return x + inputs

class MlpResNet2(nn.Module):
    def __init__(self, num_outputs):
        super(MlpResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn0 = GraphBatchNorm(num_outputs)
        self.fc0 = GraphConv1x1(num_outputs, num_outputs, batch_norm=None)
        self.bn1 = GraphBatchNorm(num_outputs)
        self.fc1 = GraphConv1x1(num_outputs, num_outputs, batch_norm=None)

    def forward(self, L, mask, inputs):
        x = inputs
        x = self.bn0(x)
        x = F.elu(x)
        x = self.fc0(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.fc1(x)
        return x + inputs
