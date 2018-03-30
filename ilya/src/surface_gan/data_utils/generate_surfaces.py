import numpy as np
import scipy as sp
from plyfile import PlyData, PlyElement
import math

def semi_sphere(R, u, v):
    # TODO: Actually try more interesting surfaces
    u += 1
    u /= 2
    u *= -R[2]
    v *= np.pi
    x = R[0] * np.cos(v) * np.sqrt(u)
    y = R[1] * np.sin(v) * np.sqrt(u)
    z = -u
    return x, y, z

def elliptic_paraboloid(R, u, v):
    # TODO: Actually try more interesting surfaces
    x = R[0] * u
    y = R[1] * v
    z = R[2] * (u * u + v * v)
    return x, y, z

def parabolic_cylinder(R, u, v):
    # TODO: Actually try more interesting surfaces
    x = R[0] * u
    y = R[1] * v
    z = R[2] * u * u
    return x, y, z

def hyperbolic_paraboloid(R, u, v):
    # TODO: Actually try more interesting surfaces
    x = R[0] * u
    y = R[1] * v
    z = R[2] * (u * u - v * v)
    return x, y, z

def create_surface(ind):
    plydata = PlyData.read('grid.ply')

    R = np.random.uniform(0.5, 1.0, size=(3,))
    R[2] *= -1

    num_vertices = len(plydata['vertex'].data)
    num_faces = len(plydata['face'].data)
    num_inputs = 2

    surfaces = [semi_sphere, elliptic_paraboloid, parabolic_cylinder, hyperbolic_paraboloid]
    surface_fn = surfaces[np.random.randint(len(surfaces))]

    for i, vertex in enumerate(plydata['vertex'].data):
        x, y, z = surface_fn(R, vertex[0], vertex[1])
        vertex[0] = x
        vertex[1] = y
        vertex[2] = z

    plydata.write('../meshes/surface{}.ply'.format(ind))

if __name__ == "__main__":
    for i in range(10000):
        create_surface(i)
