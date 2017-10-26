import csv

class StatsLogger:
    def __init__(self, filename, headers):
        self.filename = filename + '.csv'
        self.headers = headers
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writeheader()

    def log(self, stats_dict):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writerow(stats_dict)
