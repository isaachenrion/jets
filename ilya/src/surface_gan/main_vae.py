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
import math

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')
parser.add_argument('--nz', type=int, default=16, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
args = parser.parse_args()
print(args)

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

V, A, F = torch.load('graph_train.pt')
V = V.float()
A = A.float()

A = Variable(A[0:args.batchSize])
F = Variable(F[0:args.batchSize])

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.decoder = Generator(args.nz, num_outputs=6)
        self.encoder = GraphCNNDiscriminator(V.size(1), args.nz * 2)

    def encode(self, x, A):
        h = self.encoder(x, A)
        mu, log_var = h.split(h.size(1) // 2, 1)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z, A):
        h = self.decoder(None, z)
        mu, log_var = h.split(h.size(2) // 2, 2)
        return mu, log_var

    def forward(self, x, A):
        z_mu, z_log_var = self.encode(x, A)
        z = self.reparametrize(z_mu, z_log_var)
        recon_mu, recon_log_var = self.decode(z, A)
        return recon_mu, recon_log_var, z_mu, z_log_var


def loss_function(recon_mu, recog_log_var, x, mu, log_var):
    LL = 0.5 * (math.log(2 * math.pi) + recog_log_var + (x - recon_mu).pow(2) / recog_log_var.exp())
    LL = LL.sum()

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return LL + KLD

model = VAE()
input = torch.FloatTensor(args.batchSize, V.size(1), V.size(2))
noise = torch.FloatTensor(args.batchSize, args.nz)
fixed_noise = torch.FloatTensor(args.batchSize, args.nz).normal_(0, 1)

if args.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    one, mone = one.cuda(), mone.cuda()
    input, A, label = input.cuda(), A.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (args.beta1, 0.999))

gen_iterations = 0
for epoch in range(10000):
    for i in range(100):
        batch_indices = (torch.rand(args.batchSize) * (V.size(0) - 1)).floor().long()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        model.zero_grad()
        real_cpu = V[batch_indices]

        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)

        recog_mean, recog_log_var, mu, log_var = model(input, A)
        loss = loss_function(recog_mean, recog_log_var, input, mu, log_var) / args.batchSize
        loss.backward()
        optimizer.step()

        print('[{}][{}] Loss: {}'.format(epoch, i, loss.data[0]))
    for j in range(input.size(0) // 10):
        mesh.save_as_ply(
                'recog/samples_epoch_%03d_real.ply' % (j,), input.data[j].cpu(), F.data[0].cpu())
        mesh.save_as_ply(
                'recog/samples_epoch_%03d_recon.ply' % (j,), recog_mean.data[j].cpu(), F.data[0].cpu())

    fake = model.decoder(None, fixed_noise)
    for j in range(fake.size(0) // 10):
        mesh.save_as_ply(
                'results/fake_samples_epoch_%03d_%03d.ply' % (j, epoch), fake.data[j].cpu(), F.data[0].cpu())
