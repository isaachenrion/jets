from __future__ import absolute_import

import torch
from plyfile import PlyData, PlyElement
from os import listdir
from os.path import isdir, isfile, join
import sys
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
from models import *
import pickle
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-epoch', type=int, default=1000, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--num-updates', type=int, default=1000, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--number-edges', type=int, default=8, metavar='N',
                    help='minimum number of edges per vertex (default: 8)')
parser.add_argument('--coarsening-levels', type=int, default=4, metavar='N',
                    help='number of coarsened graphs. (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--model', default="lap",
                    help='lap | dirac')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def read_data():
    mypath = "faust/data_plus"
    files = sorted([f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith("npy"))])

    print("Loading the dataset")
    pbar = pb.ProgressBar()

    sequences = []

    def load_file(seqname):
        sequence = np.load(open(mypath + "/" + seqname, 'rb'))
        new_sequence = []
        for i, frame in enumerate(sequence):
            frame['V'] = torch.from_numpy(frame['V'])
            frame['F'] = torch.from_numpy(frame['F'])
            frame['L'] = utils.sp_sparse_to_pt_sparse(frame['L'])

            frame['Di'] = utils.sp_sparse_to_pt_sparse(frame['Di'])
            frame['DiA'] = utils.sp_sparse_to_pt_sparse(frame['DiA'])
            new_sequence.append(frame)

        return new_sequence

    for seqname in pbar(files):
        sequences.append(load_file(seqname))

        #if len(sequences) == 1000:
        #    break
    return sequences

sequences = read_data()

def sample_patch(sequences, sample_info):
    for b in range(args.batch_size):
        shape, patch = sample_info[b]

        sample_patch.num_vertices = max(sample_patch.num_vertices, sequences[shape][patch]['V'].size(0))
        sample_patch.num_faces = max(sample_patch.num_faces, sequences[shape][patch]['F'].size(0))

    inputs = torch.zeros(args.batch_size, sample_patch.num_vertices, 3)
    mask = torch.zeros(args.batch_size, sample_patch.num_vertices, 1)
    targets = torch.zeros(args.batch_size, 1)

    laplacian = []

    Di = []
    DiA = []

    for b in range(args.batch_size):
        shape, patch = sample_info[b]

        num_vertices = sequences[shape][patch]['V'].size(0)
        num_faces = sequences[shape][patch]['F'].size(0)

        inputs[b, :num_vertices] = sequences[shape][patch]['V']
        mask[b, :num_vertices] = 1
        targets[b] = patch

        L = sequences[shape][patch]['L']
        laplacian.append(L)

        Di.append(sequences[shape][patch]['Di'])
        DiA.append(sequences[shape][patch]['DiA'])

    laplacian = utils.sparse_cat(laplacian, sample_patch.num_vertices, sample_patch.num_vertices)
    Di = utils.sparse_cat(Di, 4 * sample_patch.num_faces, 4 * sample_patch.num_vertices)
    DiA = utils.sparse_cat(DiA, 4 * sample_patch.num_vertices, 4 * sample_patch.num_faces)

    if args.cuda:
        return Variable(inputs).cuda(), Variable(targets).cuda(), Variable(mask).cuda(), Variable(laplacian).cuda(), Variable(Di).cuda(), Variable(DiA).cuda()
    else:
        return Variable(inputs), Variable(targets), Variable(mask), Variable(laplacian), Variable(Di), Variable(DiA)

sample_patch.num_vertices = 0
sample_patch.num_faces = 0

if args.model == "lap":
    model = Model()
else:
    model = DirModel()

num_params = 0
for param in model.parameters():
    num_params += param.numel()
print("Num parameters {}".format(num_params))

early_optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
late_optimizer = optim.SGD(model.parameters(), 1e-3, weight_decay=1e-5, momentum=0.9)


class DecisionNetwork(nn.Module):
    def __init__(self, num_inputs):
        super(DecisionNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs * 2, 256)
        self.do1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 256)
        self.do2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.do2(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

mlp = DecisionNetwork(128)

if args.cuda:
    model.cuda()
    mlp.cuda()


def main():
    for epoch in range(args.num_epoch):
        #torch.save(model, 'models/{}_conv.pt'.format(args.model))

        pbar = pb.ProgressBar()
        model.train()
        mlp.train()
        loss_value = 0
        accuracy_value = 0
        # Train
        for j in pbar(range(args.num_updates)):
            sample_info = []
            for i in range(args.batch_size):
                sample_info.append([0, np.random.randint(len(sequences[0]))])

            inputs, targets, mask, laplacian, Di, DiA = sample_patch(sequences, sample_info)

            if args.model == 'lap':
                outputs0 = model(inputs, laplacian, mask)
            else:
                outputs0 = model(inputs, laplacian, Di, DiA, mask)

            for i in range(args.batch_size):
                sample_info[i][0] = np.random.randint(1, 8) #len(sequences))

                if i % 2 == 0:
                    targets[i] = 0

                    while True:
                        new_patch = np.random.randint(len(sequences[sample_info[i][0]]))
                        if new_patch != sample_info[i][1]:
                            break

                    sample_info[i][1] = new_patch
                else:
                    targets[i] = 1

            inputs, _, mask, laplacian, Di, DiA = sample_patch(sequences, sample_info)

            if args.model == 'lap':
                outputs1 = model(inputs, laplacian, mask)
            else:
                outputs1 = model(inputs, laplacian, Di, DiA, mask)

            output = mlp(torch.cat([outputs0, outputs1], 1))

            accuracy_value += (output.data.round() == targets.data.round()).float().mean()
            loss = F.binary_cross_entropy(output, targets)

            if epoch < 50:
                early_optimizer.zero_grad()
                loss.backward()
                early_optimizer.step()
            else:
                late_optimizer.zero_grad()
                loss.backward()
                late_optimizer.step()

            loss_value += loss.data[0]

        print("Train epoch {}, loss {}, acc {}".format(
            epoch, loss_value / args.num_updates, accuracy_value / args.num_updates))

        if epoch > 50 and epoch % 10 == 0:
            for param_group in late_optimizer.param_groups:
                param_group['lr'] *= 0.5

        model.eval()
        mlp.eval()
        loss_value = 0
        accuracy_value = 0
        pbar = pb.ProgressBar()

        # Evaluate
        test_trials = args.num_updates // 10
        for j in pbar(range(test_trials)):
            sample_info = []
            for i in range(args.batch_size):
                sample_info.append([0, np.random.randint(len(sequences[0]))])

            inputs, targets, mask, laplacian, Di, DiA = sample_patch(sequences, sample_info)

            if args.model == 'lap':
                outputs0 = model(inputs, laplacian, mask)
            else:
                outputs0 = model(inputs, laplacian, Di, DiA, mask)

            for i in range(args.batch_size):
                sample_info[i][0] = np.random.randint(8, len(sequences))

                if i % 2 == 0:
                    targets[i] = 0

                    while True:
                        new_patch = np.random.randint(len(sequences[sample_info[i][0]]))
                        if new_patch != sample_info[i][1]:
                            break

                    sample_info[i][1] = new_patch
                else:
                    targets[i] = 1

            inputs, _, mask, laplacian, Di, DiA = sample_patch(sequences, sample_info)

            if args.model == 'lap':
                outputs1 = model(inputs, laplacian, mask)
            else:
                outputs1 = model(inputs, laplacian, Di, DiA, mask)

            output = mlp(torch.cat([outputs0, outputs1], 1))

            accuracy_value += (output.data.round() == targets.data.round()).float().mean()
            loss = F.binary_cross_entropy(output, targets)
            loss.backward()

            loss_value += loss.data[0]

        print("Test epoch {}, loss {}, acc {}".format(epoch, loss_value / test_trials, accuracy_value / test_trials))

if __name__ == "__main__":
    main()
