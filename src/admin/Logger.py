import csv
import os
import logging
from collections import OrderedDict

from src.visualizing.plot_training_stats import plot_training_versus_validation_stat,read_training_stats
from src.admin.MonitorCollection import MonitorCollection

class Logger:
    def __init__(self, directory, train, monitor_collections):
        self.train = train

        self.statsdir = os.path.join(directory, 'stats')
        self.plotsdir = os.path.join(directory, 'plots')
        self.comparison_dir = os.path.join(self.plotsdir, 'train_vs_valid')

        if not os.path.exists(self.statsdir):
            os.makedirs(self.statsdir)
        if not os.path.exists(self.plotsdir):
            os.makedirs(self.plotsdir)
        if not os.path.exists(self.comparison_dir):
            os.makedirs(self.comparison_dir)

        self.scalar_filename = os.path.join(self.statsdir, 'stats.csv')
        self.monitor_collections= monitor_collections
        self.headers = []
        for mc_name, mc in self.monitor_collections.items():
            mc.initialize(os.path.join(self.statsdir, mc_name), os.path.join(self.plotsdir, mc_name))
            for name in mc.scalar_names:
                self.headers.append(mc_name + '_' + name)
        #import ipdb; ipdb.set_trace()
        self.visualized_scalar_names = [mc.name + '_' + name for mc in self.monitor_collections.values() for name in mc.scalar_names]
        #self.headers = headers

        if self.train:
            self.shared_names = list(
                set(self.monitor_collections['dummy_train'].names) \
                & set(self.monitor_collections['valid'].names)
                )
            self.shared_scalar_names = list(
                set(self.monitor_collections['dummy_train'].scalar_names) \
                & set(self.monitor_collections['valid'].scalar_names)
                )

        with open(self.scalar_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writeheader()

    def log_scalars(self, stats_dict):
        #import ipdb; ipdb.set_trace()
        with open(self.scalar_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, self.headers)
            writer.writerow({k:v for k,v in stats_dict.items() if k in self.headers})
        if self.train:
            #plot_training_stats(self.scalar_filename, self.visualized_scalar_names, self.plotsdir)
            stats_dict = read_training_stats(self.scalar_filename, [y for x in self.shared_scalar_names for y in ['dummy_train_'+x, 'valid_'+x]])
            for name in self.shared_scalar_names:
                plot_training_versus_validation_stat(name, stats_dict['dummy_train_'+name], stats_dict['valid_'+name], self.comparison_dir)

    def log(self, compute_monitors=False,**kwargs):
        #if compute_monitors:
        #    stats_dict = {}
        #    for mc in self.monitor_collections.values():
        #        sub_stats_dict = mc(**kwargs[mc.name])
        #        sub_stats_dict = {mc.name + '_' + name: value for name, value in sub_stats_dict.items()}
        #        stats_dict.update(**sub_stats_dict)
        #        mc.visualize()
        #else:
        stats_dict = {}
        for mc_name, mc in self.monitor_collections.items():
            sub_stats_dict = kwargs[mc_name]
            sub_stats_dict = {mc_name + '_' + name: value for name, value in sub_stats_dict.items()}
            stats_dict.update(**sub_stats_dict)
            mc.visualize()
        self.log_scalars(stats_dict)
        return '\n'.join(mc.string for mc in self.monitor_collections.values())
