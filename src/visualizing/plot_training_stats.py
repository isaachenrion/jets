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

    stats_dict = convert_strings_to_numbers(stats_dict)
    return stats_dict

def convert_strings_to_numbers(stats_dict):
    for k, v in stats_dict.items():
        stats_dict[k] = [float(x) for x in v if is_number(x)]
    return stats_dict

def plot_training_versus_validation_stat(name, training_stat, validation_stat, plotsdir):
    assert len(training_stat) == len(validation_stat)

    # smoothing
    training_stat = exponential_moving_average(training_stat, 0.5)
    validation_stat = exponential_moving_average(validation_stat, 0.5)

    # plotting
    fig, ax = plt.subplots()
    x_range = np.array([int(i) for i in range(len(training_stat))])
    line, = plt.plot(x_range, training_stat, label='Training', color='blue')
    line, = plt.plot(x_range, validation_stat, label='Validation', color='red')
    if len(x_range) < 10:
        ax.set_xticks(x_range)
    else:
        ax.set_xticks([i for i in range(len(training_stat)) if i % 10 == 0])
    ax.set_yticks(np.linspace(min(min(training_stat), min(validation_stat)), max(max(training_stat), max(validation_stat)), num=10))
    plt.grid()

    # labelling
    plt.suptitle('Training versus validation {}'.format(name))
    plt.xlabel("Epochs")
    plt.ylabel("{}".format(name))
    plt.legend(loc="best")

    imgdir = plotsdir
    pkldir = os.path.join(plotsdir, 'pkl')
    image_and_pickle(fig, '{}'.format(name), imgdir, pkldir)
    plt.close(fig)
