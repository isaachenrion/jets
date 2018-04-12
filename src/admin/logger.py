import csv
import os
import logging
from collections import OrderedDict

from src.visualizing.plot_training_stats import plot_training_stats
from src.admin.MonitorCollection import MonitorCollection

class Logger:
    def __init__(self, directory, monitor_dict, train):
        self.train = train

        self.statsdir = os.path.join(directory, 'stats')
        self.plotsdir = os.path.join(directory, 'plots')

        if not os.path.exists(self.statsdir):
            os.makedirs(self.statsdir)
        if not os.path.exists(self.plotsdir):
            os.makedirs(self.plotsdir)

        self.scalar_filename = os.path.join(self.statsdir, 'scalars.csv')

        self.monitor_collection = MonitorCollection(**monitor_dict)
        self.monitor_collection.initialize(self.statsdir, self.plotsdir)

        self.headers = self.monitor_collection.scalar_monitor_names

        with open(self.scalar_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writeheader()

    def log_scalars(self, stats_dict):
        with open(self.scalar_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writerow({k:v for k,v in stats_dict.items() if k in self.headers})
        if self.train:
            plot_training_stats(self.scalar_filename, self.monitor_collection.visualized_scalar_monitor_names, self.plotsdir)

    def log(self, compute_monitors=True,**kwargs):
        if compute_monitors:
            stats_dict = self.monitor_collection(**kwargs)
            self.monitor_collection.visualize()
        else:
            stats_dict = kwargs
        self.log_scalars(stats_dict)
