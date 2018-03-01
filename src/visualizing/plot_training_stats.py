import os
import matplotlib
import matplotlib.pyplot as plt

from scipy import interp

import numpy as np
import csv

from .utils import exponential_moving_average
from .utils import image_and_pickle
from .line_graph import line_graph
from .utils import is_number

def read_training_stats(csv_filename, stats_names):
    with open(csv_filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        stats_dict = {name: [] for name in stats_names}
        for row in reader:
            for name in stats_names:
                stats_dict[name].append(row[name])
    return stats_dict

def convert_strings_to_numbers(stats_dict):
    for k, v in stats_dict.items():
        stats_dict[k] = [float(x) for x in v if is_number(x)]
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
    ax.set_yticks(np.linspace(0.2, 0.5, num=7))
    plt.grid()

    # labelling
    plt.suptitle('Training versus validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")

    imgdir = plotsdir
    pkldir = os.path.join(plotsdir, 'pkl')
    image_and_pickle(fig, 'tv_curves', imgdir, pkldir)
    plt.close(fig)


def plot_one_training_stat(name, values, plotsdir, **kwargs):
    line_graph(values, name, plotsdir, smoothing=0.7, xname='Epochs', **kwargs)

def plot_training_stats(csv_filename, stats_names, plotsdir):
    stats_dict = read_training_stats(csv_filename, stats_names)
    stats_dict = convert_strings_to_numbers(stats_dict)
    plot_training_stats_dict(stats_dict, plotsdir)
