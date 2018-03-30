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
from multiprocessing.pool import Pool

if __name__ == "__main__":
    mypath = "./np_data/"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('npy')]

    def process(seqname):
        print(seqname)
        seqpath = mypath + seqname

        Qs, Vs, Fs = np.load(seqpath)
        Ls = []
        As = []
        Ws = []

        for i, (Q, V, F) in enumerate(zip(Qs, Vs, Fs)):
            dist = mesh.dist(V, F)
            A = sp.sparse.csr_matrix(mesh.adjacency_matrix_from_faces(F, V.shape[0]))
            As.append(A)

            W = mesh.uniform_weights(dist)
            Ws.append(W)

            L = graph.laplacian(W)
            L = graph.rescale_L(L)
            Ls.append(L)

        np.save('np_data_plus/{}'.format(seqname), (Qs, Vs, As, Fs, Ls))

    pool = Pool()
    pool.map(process, files)
