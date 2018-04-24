import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .color_dict import COLOR_DICT

def visualize_spatial(spatial, string_sequence):
    color_sequence = list(map(lambda x: [y/255.0 for y in COLOR_DICT[x]], string_sequence))

    x = spatial[:,0]
    y = spatial[:,1]
    z = spatial[:,2]

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, colors=color_sequence)
    lc.set_linewidth(3)

    print('{}'.format(string_sequence))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))
    ax.add_collection3d(lc, zs=z, zdir='z')
