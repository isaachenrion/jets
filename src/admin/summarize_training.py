import os
import csv
import numpy as np

from src.admin.emailer import get_emailer

def get_scalars_csv_filename(model_dir):
    return os.path.join(model_dir, 'stats', 'scalars.csv')

def summarize_training(jobsdir, email=None, many_jobs=False, verbose=False):
    if many_jobs:
        for jobdir in os.listdir(jobsdir):
            summarize_one_job_training(os.path.join(jobsdir, jobdir), email, verbose)
    else:
        summarize_one_job_training(jobsdir, email, verbose)

def summarize_one_job_training(jobdir, email=None, verbose=False):
    csv_filenames = []
    for run in os.listdir(jobdir):
        csv_filename = os.path.join(jobdir, get_scalars_csv_filename(run))
        if os.path.isdir(os.path.join(jobdir, run)):
            csv_filenames.append(csv_filename)


    # collect final lines of training stats
    for i, csv_filename in enumerate(csv_filenames):
        with open(csv_filename, 'r', newline='') as f:
            reader = f.readlines()

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
            mean_stat = None
            std_stat = None
        aggregate_stats_dict[name] = mean_stat, std_stat

    # make out string
    out_str = ''
    try:
        with open(os.path.join(jobdir, 'command.txt'), 'r') as f:
            command_str = f.read()

    except FileNotFoundError:
        command_str = "Command not found"
    out_str += command_str


    for name in sorted(headers):
        (mean, std) = aggregate_stats_dict[name]
        if mean is not None:
            # first print the aggregated stats
            out_str += '\n********** {} **********\n'.format(name)
            out_str += '\nmean = {:.5f}\nstd = {:.5f}\n'.format(mean, std)
            # then the collected stats
            out_str += '\n'
            for s in stats_dict[name]:
                try:
                    s = float(s)
                    out_str += '{:.5f}\n'.format(s)
                except (ValueError, TypeError):
                    out_str += '{}\n'.format(s)

    # pretty print to results file
    statsfile = os.path.join(jobdir, 'stats.txt')
    with open(statsfile, 'w') as f:
        f.write(out_str)

    # send email
    if verbose:
        print(out_str)
    if email is not None:
        emailer = get_emailer(email)
        headline = aggregate_stats_dict.get('valid_save', [-123456789])[0]
        emailer.send_msg(out_str, '{:.2f} ({}: training stats)'.format(headline, jobdir.split('/')[-1]), [])
        if verbose:
            print('Emailed: {}'.format(jobdir))
