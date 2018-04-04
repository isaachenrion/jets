import torch
from plyfile import PlyData, PlyElement
from os import listdir
from os.path import isfile, join
import sys
sys.path.append('..')
import utils.graph as graph
import utils.mesh as mesh
import numpy as np

if __name__ == "__main__":

    mypath = './meshes/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    Vs = []
    As = []
    Fs = []
    for filename in files:
        plydata = PlyData.read(mypath + filename)
        V, F = mesh.ply_to_numpy(plydata)
        A = mesh.adjacency_matrix_from_faces(F, V.shape[0])

        Vs.append(V)
        As.append(A)
        Fs.append(F)

    V = torch.from_numpy(np.stack(Vs))
    A = torch.from_numpy(np.stack(As))
    F = torch.from_numpy(np.stack(Fs))

    torch.save((V, A, F), 'graph_train.pt')

    """
    num_vertices = len(plydata['vertex'].data)
    num_faces = len(plydata['face'].data)
    num_inputs = 2

    for i, vertex in enumerate(plydata['vertex'].data):
        uv = torch.Tensor(1, 2)
        uv[0, 0] = float(vertex[0])
        uv[0, 1] = float(vertex[1])

        xyz = generator(Variable(uv), Variable(z))

        vertex[0] = xyz.data[0, 0]
        vertex[1] = xyz.data[0, 1]
        vertex[2] = xyz.data[0, 2]

    plydata.write('./result.ply')
    """
