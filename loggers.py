import csv
class StatsLogger:
    def __init__(self, filename, headers):
        self.filename = filename + '.csv'
        self.headers = headers
        self.f = open(self.filename, 'a', newline='')
        self.writer = csv.DictWriter(self.f, self.headers)
        self.writer.writeheader()

    def log(self, stats_dict):
        self.writer.writerow(stats_dict)

    def close(self):
        self.f.close()
