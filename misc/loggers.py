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

    def log_scalars(self, **kwargs):
        stats_dict = {}
        for name, monitor in self.monitors.items():
            monitor_value = monitor(**kwargs)
            if monitor.scalar:
                stats_dict[name] = monitor_value
        with open(self.scalar_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writerow(stats_dict)

    def log(self, **kwargs):
        self.log_scalars(**kwargs)
