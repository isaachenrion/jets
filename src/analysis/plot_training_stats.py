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
    tl = stats_dict['train_loss']
    vl = stats_dict['valid_loss']
    plot_training_versus_validation_loss(tl, vl, plotsdir)

def plot_training_versus_validation_loss(training_loss, validation_loss, plotsdir):
    assert len(training_loss) == len(validation_loss)
    plt.figure()
    x_range = np.array([int(i) for i in range(len(training_loss))])
    line, = plt.plot(x_range, training_loss, label='Training', color='blue')
    line, = plt.plot(x_range, validation_loss, label='Validation', color='red')
    #plt.plot(x_range, training_loss, color='blue')
    #plt.plot(x_range, validation_loss, color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.grid()

    filename = os.path.join(plotsdir, 'training_curves')
    plt.savefig(filename)

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
