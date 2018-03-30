from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from generator import Generator
from discriminator import GraphCNNDiscriminator
import sys
sys.path.append('../..')
import utils.mesh as mesh

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')
parser.add_argument('--nz', type=int, default=16, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
args = parser.parse_args()
print(args)

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

criterion = nn.MSELoss()
V, A, F = torch.load('graph_train.pt')
V = V.float()
A = A.float()

A = Variable(A[0:args.batchSize])
F = Variable(F[0:args.batchSize])

netG = Generator(args.nz)
netD = GraphCNNDiscriminator(V.size(1))

input = torch.FloatTensor(args.batchSize, V.size(1), V.size(2))
noise = torch.FloatTensor(args.batchSize, args.nz)
fixed_noise = torch.FloatTensor(args.batchSize, args.nz).normal_(0, 1)
label = torch.FloatTensor(args.batchSize)
real_label = 1
fake_label = 0
one = torch.FloatTensor([1])
mone = one * -1

if args.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    one, mone = one.cuda(), mone.cuda()
    input, A, label = input.cuda(), A.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = args.lr, betas = (args.beta1, 0.999))

uv = torch.stack([torch.linspace(-1, 1, steps=5).unsqueeze(0).expand(5, 5),
                  torch.linspace(-1, 1, steps=5).unsqueeze(1).expand(5, 5)], 2)
uv = uv.view(25, 2)
uv = Variable(uv.unsqueeze(0).expand(args.batchSize, 25, 2).contiguous())

gen_iterations = 0
for epoch in range(1000):
    for i in range(1000):
        batch_indices = (torch.rand(args.batchSize) * (V.size(0) - 1)).floor().long()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = V[batch_indices]

        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label)
        output = netD(input, A)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.data.resize_(batch_size, args.nz)
        noise.data.normal_(0, 1)

        fake = netG(uv, noise)
        label.data.fill_(fake_label)
        output = netD(fake.detach(), A)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        noise.data.resize_(batch_size, args.nz)
        noise.data.normal_(0, 1)
        netG.zero_grad()
        label.data.fill_(real_label)
        output = netD(fake, A)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, 10, i, 10,
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

        if i % 100 == 0:
            fake = netG(uv, fixed_noise)
            for j in range(fake.size(0) // 10):
                mesh.save_as_ply(
                        'results/fake_samples_epoch_%03d_%03d.ply' % (j, epoch), fake.data[j].cpu(), F.data[0].cpu())
