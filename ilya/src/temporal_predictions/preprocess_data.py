import torch
from plyfile import PlyData, PlyElement
from os import listdir
from os.path import isdir, isfile, join
import sys
sys.path.append('..')
import utils.graph as graph
import utils.mesh as mesh
import numpy as np
import scipy as sp


def load_obj(filename):
    V = []  # vertex
    F = []  # face indexies

    fh = open(filename)
    for line in fh:
        if line[0] == '#':
            continue

        line = line.strip().split(' ')
        if line[0] == 'v':  # vertex
            V.append([float(line[i+1]) for i in range(3)])
        elif line[0] == 'f':  # face
            face = line[1:]
            for i in range(0, len(face)):
                face[i] = int(face[i].split('/')[0]) - 1
            F.append(face)

    V = np.array(V)
    F = np.array(F)
    
    return V, F

if __name__ == "__main__":

    sigma2 = None
    mypath = "./data/"
    files = [f for f in listdir(mypath) if isdir(join(mypath, f))]
    for seqname in files:
        seqpath = mypath + seqname
        seqfiles = [f for f in listdir(seqpath) if isfile(join(seqpath, f))]

        Qs = []
        Vs = []
        As = []
        Fs = []
        Ls = []

        indices = sorted([int(a.split('.')[0]) for a in seqfiles])
        max_index = indices[-1]

        for i in range(1, max_index+1):
            
            npy_file = seqpath + "/{}.npy".format(i)
            skeletone = np.load(npy_file)
            Q = skeletone.reshape(-1)

            obj_file = seqpath + "/{}.obj".format(i)
            V, F = load_obj(obj_file)
            A = mesh.adjacency_matrix_from_faces(F, V.shape[0])
            dist = mesh.dist(V, F)

            if sigma2 is None:
                sigma2 = np.mean(dist.data)**2

            W = mesh.exp_weights(dist, sigma2)

            L = graph.laplacian(W)
            L = graph.rescale_L(L)

            A = sp.sparse.csr_matrix(A)

            Qs.append(Q)
            Vs.append(V)
            As.append(A)
            Fs.append(F)
            Ls.append(L)

        np.save('np_data/{}'.format(seqname), (Qs, Vs, As, Fs, Ls))
