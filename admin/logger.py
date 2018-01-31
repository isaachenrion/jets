import csv
import os
import visdom
import logging

class StatsLogger:
    def __init__(self, directory, monitors, visualizing):
        self.visualizing = visualizing
        if self.visualizing:
            self.viz = visdom.Visdom()
        else:
            self.viz = None

        self.statsdir = os.path.join(directory, 'stats')
        self.plotsdir = os.path.join(directory, 'plots')
        os.makedirs(self.statsdir)
        os.makedirs(self.plotsdir)
        self.scalar_filename = os.path.join(self.statsdir, 'scalars.csv')
        self.monitors = monitors

        for monitor in self.monitors.values():
            monitor.initialize(self.statsdir, self.plotsdir, self.viz)
        self.headers = [name for name, m in self.monitors.items() if m.scalar]

        with open(self.scalar_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writeheader()


    def compute_monitors(self, **kwargs):
        stats_dict = {}
        for name, monitor in self.monitors.items():
            monitor_value = monitor(**kwargs)
            if monitor.scalar:
                stats_dict[name] = monitor_value
        return stats_dict

    def log_scalars(self, stats_dict):
        with open(self.scalar_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writerow(stats_dict)

    def log(self, compute_monitors=True,**kwargs):
        if compute_monitors:
            stats_dict = self.compute_monitors(**kwargs)
            self.log_scalars(stats_dict)
        else:
            self.log_scalars(kwargs)

    def complete_logging(self):
        for monitor in self.monitors.values():
            monitor.finish()
