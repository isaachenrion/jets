import csv
import os
import logging
from collections import OrderedDict

from src.visualizing.plot_training_stats import plot_training_stats
from src.admin.MonitorCollection import MonitorCollection

class Logger:
    def __init__(self, directory, train, monitor_collections):
        self.train = train

        self.statsdir = os.path.join(directory, 'stats')
        self.plotsdir = os.path.join(directory, 'plots')

        if not os.path.exists(self.statsdir):
            os.makedirs(self.statsdir)
        if not os.path.exists(self.plotsdir):
            os.makedirs(self.plotsdir)

        self.scalar_filename = os.path.join(self.statsdir, 'scalars.csv')

        self.monitor_collections = monitor_collections.values()
        for mc in self.monitor_collections:
            mc.initialize(self.statsdir, os.path.join(self.plotsdir , mc.name))

        self.headers = [mc.name + '_' + name for mc in self.monitor_collections for name in mc.scalar_monitor_names]
        self.visualized_scalar_monitor_names = [mc.name + '_' + name for mc in self.monitor_collections for name in mc.scalar_monitor_names]

        with open(self.scalar_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writeheader()

    def log_scalars(self, stats_dict):
        with open(self.scalar_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writerow({k:v for k,v in stats_dict.items() if k in self.headers})
        if self.train:
            plot_training_stats(self.scalar_filename, self.visualized_scalar_monitor_names, self.plotsdir)

    def log(self, compute_monitors=True,**kwargs):
        if compute_monitors:
            stats_dict = {}
            for mc in self.monitor_collections:
                sub_stats_dict = mc(**kwargs[mc.name])
                sub_stats_dict = {mc.name + '_' + name: value for name, value in sub_stats_dict.items()}
                stats_dict.update(**sub_stats_dict)
                mc.visualize()
        else:
            stats_dict = kwargs
        self.log_scalars(stats_dict)
        return '\n'.join(mc.string for mc in self.monitor_collections)
