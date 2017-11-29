import csv
import os
class StatsLogger:
    def __init__(self, directory, monitors):
        self.directory = directory
        os.makedirs(directory)
        self.scalar_filename = os.path.join(directory, 'scalars.csv')

        self.monitors = monitors
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
