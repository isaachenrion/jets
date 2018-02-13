import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6, 6)

from scipy import interp

import numpy as np
import csv

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

def plot_training_stats(csv_filename, stats_names, plotsdir):
    stats_dict = read_training_stats(csv_filename, stats_names)
    plot_training_stats_dict(stats_dict, plotsdir)
