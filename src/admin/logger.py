import csv
import os
import logging
from collections import OrderedDict

from ..visualizing.plot_training_stats import plot_training_stats


class Logger:
    def __init__(self, directory, monitor_dict, visualizing, train):
        self.train = train
        self.visualizing = visualizing

        self.statsdir = os.path.join(directory, 'stats')
        self.plotsdir = os.path.join(directory, 'plots')
        os.makedirs(self.statsdir)
        os.makedirs(self.plotsdir)
        self.scalar_filename = os.path.join(self.statsdir, 'scalars.csv')

        self.monitors = OrderedDict()
        self.visualized_scalar_monitors = []
        self.headers = []
        self.add_many_monitors(**monitor_dict)

        with open(self.scalar_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writeheader()

        #import ipdb; ipdb.set_trace()

    def add_many_monitors(self, **monitors):
        for name, monitor in monitors.items():
            self.add_monitor(name, monitor)

    def add_monitor(self, name, monitor):
        self.monitors[name] = monitor
        if monitor.scalar and monitor.visualizing:
            self.visualized_scalar_monitors.append(name)
        if monitor.scalar:
            self.headers.append(name)
        monitor.initialize(self.statsdir, self.plotsdir)
        return monitor

    def compute_monitors(self, **kwargs):
        stats_dict = {}
        for name, monitor in self.monitors.items():
            monitor_value = monitor(**kwargs)
            if monitor.scalar:
                stats_dict[name] = monitor_value
            if monitor.visualizing:
                monitor.visualize()
        return stats_dict

    def log_scalars(self, stats_dict):
        with open(self.scalar_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writerow(stats_dict)
        if self.train:
            ##pass
            plot_training_stats(self.scalar_filename, self.visualized_scalar_monitors, self.plotsdir)

    def log(self, compute_monitors=True,**kwargs):
        if compute_monitors:
            stats_dict = self.compute_monitors(**kwargs)
            self.log_scalars(stats_dict)
        else:
            self.log_scalars(kwargs)

    def complete_logging(self):
        for monitor in self.monitors.values():
            monitor.finish()
