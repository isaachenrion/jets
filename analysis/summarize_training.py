import os
import csv
import numpy as np

from admin.emailer import get_emailer

def get_scalars_csv_filename(model_dir):
    return os.path.join(model_dir, 'stats', 'scalars.csv')

def summarize_training(jobdir, email=False):
    csv_filenames = []
    for run in os.listdir(jobdir):
        csv_filename = os.path.join(jobdir, get_scalars_csv_filename(run))
        if os.path.isdir(os.path.join(jobdir, run)):
            csv_filenames.append(csv_filename)

    # collect final lines of training stats
    for i, csv_filename in enumerate(csv_filenames):
        with open(csv_filename, 'r', newline='') as f:
            #reader = csv.DictReader(f)
            reader = f.readlines()

            #for i, row in enumerate(reader)
            if i == 0:
                headers = reader[0].strip().split(',')
                stats_dict = {name: [] for name in headers}
            for j, name in enumerate(headers):
                stats_dict[name].append(reader[-1].strip().split(',')[j])

    # compute aggregate stats
    aggregate_stats_dict = {}
    for name in headers:
        try:
            arr = np.array(stats_dict[name]).astype(float)
            mean_stat = np.mean(arr)
            std_stat = np.std(arr)
        except (TypeError, ValueError) as e:
            #print(e, name, stats_dict[name])
            mean_stat = None
            std_stat = None
        aggregate_stats_dict[name] = mean_stat, std_stat

    # make out string
    out_str = ''
    for name in sorted(headers):
        (mean, std) = aggregate_stats_dict[name]
        if mean is not None:
            # first print the aggregated stats
            out_str += '\n********** {} **********\n'.format(name)
            out_str += '\nmean = {:.2f}\nstd = {:.2f}\n'.format(mean, std)
            # then the collected stats
            out_str += '\n'
            for s in stats_dict[name]:
                try:
                    s = float(s)
                    out_str += '{:.2f}\n'.format(s)
                except (ValueError, TypeError):
                    out_str += '{}\n'.format(s)

    # pretty print to results file
    statsfile = os.path.join(jobdir, 'stats.txt')
    with open(statsfile, 'w') as f:
        f.write(out_str)

    # send email
    if email:
        print('Emailing')
        emailer = get_emailer()
        emailer.send_msg(out_str, '{}: training stats'.format(jobdir), [statsfile])
