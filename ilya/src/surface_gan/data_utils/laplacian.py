import numpy as np
import scipy as sp
from scipy import sparse
from plyfile import PlyData, PlyElement
import math
import itertools
import sys
sys.path.append('../..')
import utils.graph as graph
import utils.mesh as mesh
import time


if __name__ == "__main__":
    plydata = PlyData.read('./dolphins.ply')
    V, F = ply_to_numpy(plydata)

    for k in range(100):
        dist = mesh.dist(V, F)
        areas = mesh.area(F, dist)

        #W = mesh.uniform_weights(dist)
        W, A_inv = mesh.cotangent_weights(F, areas, dist)
        #print(W)

        L = mesh.laplacian(W, A_inv)
        #L = graph.laplacian(W, symmetric=False)

        V = V - 0.1 * L * V
        for i in range(len(plydata['vertex'].data)):
            plydata['vertex'].data['x'][i] = V[i][0]
            plydata['vertex'].data['y'][i] = V[i][1]
            plydata['vertex'].data['z'][i] = V[i][2]
        plydata.write('results/{}.ply'.format(k))
