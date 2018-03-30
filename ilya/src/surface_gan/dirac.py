import numpy as np
import scipy as sp
from plyfile import PlyData, PlyElement
import math

def triangle_area(e):
    lengths = np.zeros([3])
    for i in range(3):
        j = (i + 1) % 3
        vec = np.array([e[i][0] - e[j][0], e[i][1] - e[j][1], e[i][2] - e[j][2]])
        lengths[i] = np.linalg.norm(vec)

    length_sum = np.sum(lengths) / 2
    return math.sqrt(length_sum * (length_sum - lengths[0]) * (length_sum - lengths[1]) * (length_sum - lengths[2]))

def quaternion_matrix(x):
    a, b, c, d = x.tolist()
    return np.array([[a, -b, -c, -d],
                     [b,  a, -d,  c],
                     [c,  d,  a, -b],
                     [d, -c,  b,  a]])

if __name__ == "__main__":
    plydata = PlyData.read('../meshes/cube.ply')

    num_vertices = len(plydata['vertex'].data)
    num_faces = len(plydata['face'].data)
    num_inputs = 2

    for i, vertex in enumerate(plydata['vertex'].data):
        print("Vertex {}: {}".format(i, vertex))

    for i, face in enumerate(plydata['face'].data):
        print("Face {}: {}".format(i, face))

    # Dirac operator
    D = np.zeros([4 * num_faces, 4 * num_vertices])

    for i in range(num_faces):
        face_vertices = plydata['face'].data[i][0]
        e = [plydata['vertex'].data[0],
             plydata['vertex'].data[1],
             plydata['vertex'].data[2]]

        A = triangle_area(e)

        for ind, j in enumerate(face_vertices):
            ind1 = face_vertices[(ind + 1) % 3]
            ind2 = face_vertices[(ind + 2) % 3]

            e1 = plydata['vertex'].data[ind1]
            e2 = plydata['vertex'].data[ind2]

            e = np.array([0, e1[0] - e2[0], e1[1] - e2[1], e1[2] - e2[2]])

            D[i * 4:(i + 1) * 4, j * 4: (j + 1) *
              4] = -quaternion_matrix(e) / A

    # signal
    x = np.random.randn(num_vertices, num_inputs)

    # signal in quaternions
    q_x = np.zeros((4 * num_vertices, 4 * num_inputs))
    for i in range(num_vertices):
        for j in range(num_inputs):
            quaternion = np.zeros((4))
            quaternion[0] = x[i, j]
            q_x[4*i:4*(i+1), 4*j:4*(j+1)] = quaternion_matrix(quaternion)

    D_x = D @ q_x

    print("Signal at the first face:")
    print(D_x[16:20, 4:8])
    print(D_x.shape)
