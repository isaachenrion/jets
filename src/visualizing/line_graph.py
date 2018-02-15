import os
import matplotlib.pyplot as plt
import numpy as np

from .utils import image_and_pickle
from .utils import exponential_moving_average

def line_graph(values, filename, plotsdir, smoothing=None, title='', xname='', yname='', color='blue'):
    if smoothing is not None:
        values = exponential_moving_average(values, smoothing)
    fig, ax = plt.subplots()
    x_range = np.array([int(i) for i in range(len(values))])
    line, = plt.plot(x_range, values, color=color)
    #if len(x_range) < 10:
    #    ax.set_xticks(x_range)
    #else:
    #    ax.set_xticks([i for i in range(len(training_loss)) if i % 10 == 0])
    #ax.set_yticks(np.linspace(0.0, 1.0, num=21))
    plt.grid()
    # labelling
    plt.suptitle(title)
    plt.xlabel(xname)
    plt.ylabel(yname)
    #plt.legend(loc="best")
    imgdir = plotsdir
    pkldir = os.path.join(plotsdir, 'pkl')
    image_and_pickle(fig, filename, imgdir, pkldir)
    plt.close(fig)
