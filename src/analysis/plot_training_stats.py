import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
plt.rcParams["figure.figsize"] = (6, 6)

from scipy import interp

import numpy as np
import csv

from .utils import exponential_moving_average
from .plotting import image_and_pickle

def read_training_stats(csv_filename, stats_names):
    with open(csv_filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        stats_dict = {name: [] for name in stats_names}
        for row in reader:
            for name in stats_names:
                stats_dict[name].append(row[name])
    return stats_dict

def plot_training_stats_dict(stats_dict, plotsdir):
    for k, v in stats_dict.items():
        plot_one_training_stat(k, v, plotsdir)
    tl = [float(x) for x in stats_dict['train_loss']]
    vl = [float(x) for x in stats_dict['valid_loss']]
    plot_training_versus_validation_loss(tl, vl, plotsdir)

def plot_training_versus_validation_loss(training_loss, validation_loss, plotsdir):
    assert len(training_loss) == len(validation_loss)
    # smoothing
    training_loss = exponential_moving_average(training_loss, 0.5)
    validation_loss = exponential_moving_average(validation_loss, 0.5)

    # plotting
    fig, ax = plt.subplots()
    x_range = np.array([int(i) for i in range(len(training_loss))])
    line, = plt.plot(x_range, training_loss, label='Training', color='blue')
    line, = plt.plot(x_range, validation_loss, label='Validation', color='red')
    if len(x_range) < 10:
        ax.set_xticks(x_range)
    else:
        ax.set_xticks([i for i in range(len(training_loss)) if i % 10 == 0])
    ax.set_yticks(np.linspace(0.0, 1.0, num=21))
    plt.grid()

    # labelling
    plt.suptitle('Training versus validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")

    imgdir = plotsdir
    pkldir = os.path.join(plotsdir, 'pkl')
    if not os.path.exists(pkldir):
        os.makedirs(pkldir)
    image_and_pickle(fig, 'tv_curves', imgdir, pkldir)


def plot_one_training_stat(name, values, plotsdir):
    plt.figure()
    x_range = np.array([int(i) for i in range(len(values))])
    plt.plot(x_range, values, color='blue')
    plt.xlabel("Epochs")
    plt.ylabel(name)
    #plt.legend(loc="best")
    plt.grid()

    filename = os.path.join(plotsdir, name)
    plt.savefig(filename)
    plt.close()

def plot_training_stats(csv_filename, stats_names, plotsdir):
    stats_dict = read_training_stats(csv_filename, stats_names)
    plot_training_stats_dict(stats_dict, plotsdir)
