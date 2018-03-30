import torch
from plyfile import PlyData, PlyElement
from os import listdir
from os.path import isdir, isfile, join
import sys
sys.path.append('..')
import utils.graph as graph
import utils.mesh as mesh
import utils.utils_pt as utils
import numpy as np
import scipy as sp
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import progressbar as pb
import os
from models_pose import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-epoch', type=int, default=1000, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--num-updates', type=int, default=1000, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--k', type=int, default=1, metavar='N',
                    help='num of training epochs (default: 1)')
parser.add_argument('--number-edges', type=int, default=8, metavar='N',
                    help='minimum number of edges per vertex (default: 8)')
parser.add_argument('--coarsening-levels', type=int, default=4, metavar='N',
                    help='number of coarsened graphs. (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def read_data():
    mypath = "./np_data/"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    sequences = []
    for seqname in files:
        if seqname.endswith("npy"):
            sequence = np.load(mypath + "/" + seqname)
            sequences.append(sequence)

    return sequences

current_configs = None
current_vertices = None
adjacency = None
laplacian = None

def sample_batch(sequences, is_training):
    global current_configs
    global current_vertices
    global adjacency
    global laplacian

    if current_configs is None:
        current_configs = torch.zeros(args.batch_size, sequences[0][0][0].shape[0])
        current_vertices = torch.zeros(args.batch_size, sequences[0][1][0].shape[0], sequences[0][1][0].shape[1])
        adjacency = torch.zeros(args.batch_size, sequences[0][2][0].shape[0], sequences[0][2][0].shape[1])
        laplacian = torch.zeros(args.batch_size, sequences[0][4][0].shape[0], sequences[0][4][0].shape[1])

    for b in range(args.batch_size):
        scale = 1.0
        if is_training:
            scale = torch.rand(1)[0] + 0.5
            while True:
                ind = np.random.randint(0, len(sequences)-1)
                if ind % 5 == 0:
                    break
        else:
            while True:
                ind = np.random.randint(0, len(sequences)-1)
                if ind % 5 > 0:
                    break

        offset = np.random.randint(0, len(sequences[ind][0])-1)
        target_offset = np.random.randint(0, len(sequences[ind][0])-1)

        current_configs[b] = torch.from_numpy(sequences[ind][0][offset])
        current_vertices[b] = torch.from_numpy(sequences[ind][1][offset])

        #adjacency[b] = torch.from_numpy(sequences[ind][2][offset].todense())

        laplacian[b] = torch.from_numpy(sequences[ind][4][offset].todense())

    return Variable(current_configs), Variable(current_vertices), Variable(adjacency), Variable(laplacian)

sequences = read_data()

#model = Model(args.k)
model = MLP(sequences[0][2][0].shape[0])
model = nn.DataParallel(model)

if args.cuda:
    model.cuda()

early_optimizer = optim.Adam(model.parameters(), 1e-3)
late_optimizer = optim.SGD(model.parameters(), 1e-3, momentum=0.9)

faces = torch.from_numpy(sequences[0][3][0])

def main():
    for epoch in range(args.num_epoch):
        pbar = pb.ProgressBar()
        model.train()
        loss_value = 0
        # Train
        for j in pbar(range(args.num_updates)):
            current_configs, current_vertices, adjacency, laplacian = sample_batch(sequences, True)
            if args.cuda:
                current_configs, current_vertices, adjacency, laplacian = current_configs.cuda(), current_vertices.cuda(), adjacency.cuda(), laplacian.cuda()

            pred_config = model(current_vertices, adjacency, laplacian)
            loss = F.smooth_l1_loss(pred_config, current_configs, size_average=False) / args.batch_size
            if epoch < 10:
                early_optimizer.zero_grad()
                loss.backward()
                early_optimizer.step()
            else:
                late_optimizer.zero_grad()
                loss.backward()
                late_optimizer.step()

            loss_value += loss.data[0]

        print("Train epoch {}, loss {}".format(epoch, loss_value / args.num_updates))

        if epoch > 10 and epoch % 10 == 0:
            for param_group in late_optimizer.param_groups:
                param_group['lr'] *= 0.5

        model.eval()
        loss_value = 0
        pbar = pb.ProgressBar()

        # Evaluate
        test_trials = args.num_updates // 10
        for j in pbar(range(test_trials)):
            current_configs, current_vertices, adjacency, laplacian = sample_batch(sequences, False)
            if args.cuda:
                current_configs, current_vertices, adjacency, laplacian = current_configs.cuda(), current_vertices.cuda(), adjacency.cuda(), laplacian.cuda()

            pred_config = model(current_vertices, adjacency, laplacian)
            loss = F.smooth_l1_loss(pred_config, current_configs, size_average=False) / args.batch_size
            loss_value += loss.data[0]

        print("Test epoch {}, loss {}".format(epoch, loss_value / test_trials))

        for t in range(124):
            print(pred_config.data[0][t], current_configs.data[0][t])


main()
