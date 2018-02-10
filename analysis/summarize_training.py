import os
import csv
import numpy as np

def get_scalars_csv_filename(model_dir):
    return os.path.join(model_dir, 'stats', 'scalars.csv')

def summarize_training(jobdir):
    csv_filenames = []
    for run in os.listdir(jobdir):
        csv_filename = os.path.join(jobdir, get_scalars_csv_filename(run))
        if os.path.isdir(csv_filename):
            csv_filenames.append(csv_filename)
            
    # collect final lines of training stats
    for i, csv_filename in enumerate(csv_filenames):
        with open(csv_filename, 'r', newline='') as f:
            #reader = csv.DictReader(f)
            reader = f.readlines()

            #for i, row in enumerate(reader)
            if i == 0:
                headers = reader[0].split(',')
                stats_dict = {name: [] for name in headers}
            for j, name in enumerate(headers):
                stats_dict[name].append(reader[-1][j])

    # compute aggregate stats
    aggregate_stats_dict = {}
    for name in headers:
        try:
            mean_stat = np.mean(np.array(stats_dict[name]))
            std_stat = np.std(np.array(stats_dict[name]))
        except TypeError as e:
            print(e, name, stats_dict[name])
            mean_stat = None
            std_stat = None
        aggregate_stats_dict[name] = mean_stat, std_stat

    # pretty print to results file
    with open(os.path.join(jobdir, 'stats.txt'), 'w') as f:
        for name, (mean, std) in aggregate_stats_dict.items():
            if mean is not None:
                # first print the aggregated stats
                f.write('{}: mean = {}, std = {}\n'.format(name, mean, std))
                # then the collected stats
                f.write('Individual statistics\n')
                for s in stats_dict[name]: f.write('{}\n'.format(s))
