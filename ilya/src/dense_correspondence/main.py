import torch
import os
from os import listdir
from os.path import isdir, isfile, join
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.mesh as mesh
import utils.utils_pt as utils
import numpy as np
import scipy as sp
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
from models import *
import pickle
import time
import gc

# Training settings
parser = argparse.ArgumentParser(description='SurfaceNN dense correspondence example')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--num-epoch', type=int, default=110, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--num-updates', type=int, default=100, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--model', default="lap",
                    help='lap | dirac | avg | mlp')
parser.add_argument('--datapath', default="datapath",
                    help='datapath')
parser.add_argument('--result-prefix', default='naive')
parser.add_argument('--lr', type = float, default=1e-3)
parser.add_argument('--layer', type = int, default=15)
parser.add_argument('--loss', default='l2')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

result_identifier = f'{args.result_prefix}_{args.model}_{args.loss}_{args.layer}_{args.lr}'

def read_data(seqname):
#    mypath = args.datapath
#    files = sorted(glob.glob(mypath+'/*.npz'))

#    my_print("Loading the dataset")
    with np.load(seqname) as sequence:
        new_sequence = []
        frame = {}
        frame['V'] = torch.from_numpy(sequence['V'])
        frame['F'] = torch.from_numpy(sequence['F'])
        frame['L'] = utils.sp_sparse_to_pt_sparse(sequence['L'].item().astype('f')).coalesce()

        if 'dir' in args.model:
            frame['Di'] = utils.sp_sparse_to_pt_sparse(sequence['D'].item().astype('f')).coalesce()
            frame['DiA'] = utils.sp_sparse_to_pt_sparse(sequence['DA'].item().astype('f')).coalesce()
        else:
            frame['Di'] = None
            frame['DiA'] = None

        frame['label'] = torch.from_numpy(sequence['label'])
        frame['label_inv'] = torch.from_numpy(sequence['label_inv'])
        frame['G'] = torch.from_numpy(sequence['dist_mat'].astype('f'))

    return frame


#sequences = read_data()

test_ind = 0

def sample_batch(sequences, is_training, is_fixed=False):
    global test_ind
    indices = []
    offsets = []

    input_frames = 1
    output_frames = 40
    gc.collect()
    for b in range(args.batch_size):
        if is_training:
            test_ind = 0
            ind = np.random.randint(0, len(sequences) // 10 * 8 + 1)
            offsets.append(0)
        elif not is_fixed:
            ind = 0# len(sequences) // 10 * 8 + test_ind
            offsets.append(0)
            test_ind += 1
        elif is_fixed:
            ind = len(sequences) // 10 * 8 + b
            offsets.append(b % (len(sequence_ind) - input_frames - output_frames))

        sequence_ind = read_data(sequences[ind])

        sample_batch.num_vertices = max(sample_batch.num_vertices, sequence_ind['V'].size(0))
        sample_batch.num_faces = max(sample_batch.num_faces, sequence_ind['F'].size(0))

        indices.append(ind)

    inputs = torch.zeros(args.batch_size, sample_batch.num_vertices, 3 * input_frames)
    #targets = (torch.zeros(args.batch_size, sample_batch.num_vertices, sample_batch.num_vertices).cuda(), torch.zeros(args.batch_size, sample_batch.num_vertices).cuda(), torch.zeros(args.batch_size, sample_batch.num_vertices).cuda())
    mask = torch.zeros(args.batch_size, sample_batch.num_vertices, 1)
    faces = torch.zeros(args.batch_size, sample_batch.num_faces, 3).long()
    laplacian = []
    targets = [None]*args.batch_size

    Di = []
    DiA = []

    for b, (ind, offset) in enumerate(zip(indices, offsets)):
        num_vertices = sequence_ind['V'].size(0)
        num_faces = sequence_ind['F'].size(0)

        for i in range(input_frames):
            inputs[b, :num_vertices, 3*i:3*(i+1)] = sequence_ind['V']

        targets[b] = (sequence_ind['G'].cuda(),
                      sequence_ind['label'].cuda(),
                      sequence_ind['label_inv'].cuda())

        mask[b, :num_vertices] = 1
        faces[b, :num_faces] = sequence_ind['F']

        L = sequence_ind['L']
        laplacian.append(L)

        if 'dir' in args.model:
            Di.append(sequence_ind['Di'])
            DiA.append(sequence_ind['DiA'])

    laplacian = utils.sparse_cat(laplacian, sample_batch.num_vertices, sample_batch.num_vertices).coalesce()

    if 'dir' in args.model:
        Di = utils.sparse_cat(Di, 4 * sample_batch.num_faces, 4 * sample_batch.num_vertices).coalesce()
        DiA = utils.sparse_cat(DiA, 4 * sample_batch.num_vertices, 4 * sample_batch.num_faces).coalesce()

    if args.cuda:
        if 'dir' in args.model:
            return Variable(inputs).cuda(), targets, Variable(mask).cuda(), Variable(laplacian).cuda(), Variable(Di).cuda(), Variable(DiA).cuda(), faces
        else:
            return Variable(inputs).cuda(), (targets), Variable(mask).cuda(), Variable(laplacian).cuda(), None, None, faces
    else:
        return Variable(inputs), targets, Variable(mask), Variable(laplacian), Variable(Di), Variable(DiA), faces

sample_batch.num_vertices = 7000
sample_batch.num_faces = 0

def aggregate_batch_G(outputs, targetX, targetY):
    batch_size = outputs.size(0)
    for i in range(batch_size):
        GA, lA, liA = targetX[i]
        GB, lB, liB = targetY[i]
        NA = lA.size(0)
        NB = lB.size(0)
        G = torch.cuda.FloatTensor(outputs.size(1), outputs.size(2)).zero_()
        G[:NA, :NB] = GA[:, liA[lB]] + GB[liB[lA],:]
        listG.append(G)
    
    FullG = torch.stack(listG)
    return FullG
    
def loss_fun_l2(outputs, targetX, targetY):
    FullG = aggregate_batch_G(outputs, targetX, targetY)
    lossl2 = torch.sqrt(torch.sum((Variable(FullG) - outputs)**2))
    return lossl2

def loss_fun_sl1(outputs, targetX, targetY):
    FullG = aggregate_batch_G(outputs, targetX, targetY)
    return F.smooth_l1_loss(Variable(FullG), outputs, size_average=False) / outputs.size(0)
    
def my_print(stuff):
    print(stuff, file=sys.stderr) # also to err
    logfile = f'log/{result_identifier}.log'
    with open(logfile,'a') as fp:
        print(stuff, file=fp)
        #fp.write(stuff)


def main():
    sequences = sorted(glob.glob(args.datapath+'/*.npz'))

    model = (SiameseModel(args.model, args.layer))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()

    my_print("Num parameters {}".format(num_params))

    if args.cuda:
        model.cuda()

    early_optimizer = optim.Adam(model.parameters(),args.lr, weight_decay=1e-5)
    late_optimizer = optim.SGD(model.parameters(), 1e-3, weight_decay=1e-5, momentum=0.9)
    
    if args.loss == 'l2':
        loss_fun = loss_fun_l2
    elif args.loss == 'sl1':
        loss_fun = loss_fun_sl1
    
    for epoch in range(args.num_epoch):
        gc.collect()
        #torch.save(model, 'models/{}_conv.pt'.format(args.model))

        model.train()
        loss_value = 0
        # Train
        for j in (range(args.num_updates)):
            inputX, targetX, maskX, laplacianX, DiX, DiAX, facesX = sample_batch(sequences, True)
            inputY, targetY, maskY, laplacianY, DiY, DiAY, facesY = sample_batch(sequences, True)

            if args.model in ["lap", "avg", "mlp"]:
                outputs = model([laplacianX, maskX],[laplacianY, maskY], inputX, inputY)
            else:
                outputs = model([DiX, DiAX, maskX],[DiY, DiAY, maskY], inputX, inputY)

            mask = torch.bmm(maskX, maskY.transpose(1,2))
            outputs = outputs * mask.expand_as(outputs)
            loss = loss_fun(outputs, targetX, targetY)/args.batch_size

            early_optimizer.zero_grad()
            loss.backward()
            early_optimizer.step()
            torch.cuda.synchronize()
            end = time.time()

            loss_value += loss.data[0]

        my_print("Train epoch {}, loss {}".format(
            epoch, loss_value / args.num_updates))
        sys.stdout.flush()

        if epoch > 50 and epoch % 10 == 0:
            for param_group in early_optimizer.param_groups:
                param_group['lr'] *= 0.5

        #model.eval()
        loss_value = 0

        # Evaluate
        test_trials = len(sequences) // 5 // args.batch_size+1
        for j in (range(test_trials)):
            inputX, targetX, maskX, laplacianX, DiX, DiAX, facesX = sample_batch(sequences, False)
            inputY, targetY, maskY, laplacianY, DiY, DiAY, facesY = sample_batch(sequences, False)

            if args.model in ["lap", "avg", "mlp"]:
                outputs = model([laplacianX, maskX],[laplacianY, maskY], inputX, inputY)
            else:
                outputs = model([DiX, DiAX, maskX],[DiY, DiAY, maskY], inputX, inputY)

            mask = torch.bmm(maskX, maskY.transpose(1,2))
            outputs = outputs * mask.expand_as(outputs)
            loss = loss_fun(outputs, targetX, targetY)/args.batch_size
            loss.backward() # because of a problem with caching

            loss_value += loss.data[0]

        my_print("Test epoch {}, loss {}".format(epoch, loss_value / test_trials))
        sys.stdout.flush()

        if epoch % 10 == 9:
            torch.save(
                model.state_dict(),
                f'pts/{result_identifier}_state.pts')

if __name__ == "__main__":
    main()
