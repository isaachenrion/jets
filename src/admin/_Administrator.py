import torch
import logging
import os
import datetime
import numpy as np
import socket
import time
import shutil
import subprocess

from collections import OrderedDict

from .utils import get_logfile
from .emailer import get_emailer
from .signal_handler import SignalHandler
from .Logger import Logger

from src.monitors import *
from ..misc.constants import RUNNING_MODELS_DIR, ALL_MODEL_DIRS

if torch.cuda.is_available():
    import GPUtil

def get_git_revision_short_hash():
    s = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    s = str(s).split('\'')[1]
    return s

class _Administrator:
    '''
    Base class for experiment administrators. This object performs a number of crucial jobs.
    1) Sets random seeds and handles GPU device
    2) Creates model directories to store results files
    3) Sets logging outputs
    4) Contains signal handler which reacts to signal events e.g. "kill", "interrupt"
    5) Contains emailer which is responsible for sending results by email
    6) Sets up the monitors, e.g. for loss, gradient norms, lr, accuracy etc.

    Function (6) is handled on a per-problem basis, therefore is not implemented.
    You should subclass this and implement setup_monitors
    '''
    def __init__(
            self,
            train=None,
            dataset=None,
            model=None,
            debug=None,
            slurm=None,
            slurm_array_task_id=None,
            slurm_array_job_id=None,
            gpu=None,
            seed=None,
            email=None,
            epochs=None,
            visualizing=None,
            silent=None,
            verbose=None,
            cmd_line_args=None,
            models_dir=None,
            **kwargs
            ):


        pid = os.getpid()
        host = socket.gethostname()


        passed_args = dict(**locals())
        kwargs = passed_args.pop('kwargs', None)
        passed_args.pop('self')
        passed_args.update(kwargs)

        self.models_dir = models_dir
        self.debug = debug
        self.slurm = slurm
        self.slurm_array_task_id = slurm_array_task_id
        self.slurm_array_job_id = slurm_array_job_id
        self.cmd_line_args = cmd_line_args
        self.pid = pid
        self.host = host
        self.train = train
        self.epochs = epochs


        self.cuda_and_random_seed(gpu, seed, passed_args)
        self.passed_args = passed_args

        self.create_all_model_dirs()
        self.setup_model_directory(dataset, model)
        self.setup_logging(silent, verbose)
        self.setup_signal_handler(email)
        self.setup_logger()
        self.record_settings(passed_args)
        self.initial_email()

    def setup_monitors(*args, **kwargs):
        raise NotImplementedError

    def cuda_and_random_seed(self, gpu, seed, passed_args):
        if gpu != "" and torch.cuda.is_available():
            torch.cuda.device(gpu)

        if seed is None:
            seed = np.random.randint(0, 2**16 - 1)
        np.random.seed(seed)

        if gpu != "" and torch.cuda.is_available():
            torch.cuda.device(gpu)
            torch.cuda.manual_seed(seed)

        else:
            torch.manual_seed(seed)

        self.seed = seed
        self.gpu = gpu

        passed_args['seed'] = self.seed
        passed_args['gpu'] = self.gpu


    def setup_model_directory(self, dataset, model):
        self.current_dir = RUNNING_MODELS_DIR
        self.root_dir = os.path.join(self.models_dir, self.current_dir)
        dt = datetime.datetime.now()
        self.start_dt = dt

        if self.slurm and self.train:
            filename_exp = '{}'.format(self.slurm_array_job_id)
            self.leaf_dir = self.slurm_array_task_id
        else:

            filename_exp = '{}-{}-{:02d}-{:02d}-{:02d}_{}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second, self.pid)
            self.leaf_dir = ''

        self.intermediate_dir = os.path.join(dataset, model, filename_exp)
        self.exp_dir = os.path.join(self.root_dir,self.intermediate_dir,self.leaf_dir)

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)


    def create_all_model_dirs(self):
        for intermediate_dir in ALL_MODEL_DIRS:
            model_dir = os.path.join(self.models_dir, intermediate_dir)
            if not os.path.isdir(model_dir):
                try:
                    os.makedirs(model_dir)
                except FileExistsError:
                    pass

    def setup_logging(self, silent, verbose):
        ''' SET UP LOGGING '''
        '''----------------------------------------------------------------------- '''
        self.logfile = get_logfile(self.exp_dir, silent, verbose)
        logging.info("running on {}".format(self.host))
        logging.info(self.exp_dir)

    def setup_signal_handler(self, email):
        ''' SIGNAL HANDLER '''
        '''----------------------------------------------------------------------- '''

        if email is not None and not self.slurm:
            self.emailer = get_emailer(email)
        else:
            self.emailer = None

        if self.slurm:
            subject_string = '{} (Machine = {}, Logfile = {}, Slurm id = {}-{}, GPU = {})'.format("[DEBUGGING] " if self.debug else "", self.host, self.logfile, self.slurm_array_job_id, self.slurm_array_task_id, self.gpu)
        else:
            subject_string = '{} (Machine = {}, Logfile = {}, PID = {}, GPU = {})'.format("[DEBUGGING] " if self.debug else "", self.host, self.logfile, self.pid, self.gpu)

        self.signal_handler = SignalHandler(
                                emailer=self.emailer,
                                logfile=self.logfile,
                                current_dir=self.current_dir,
                                models_dir=self.models_dir,
                                intermediate_dir=self.intermediate_dir,
                                leaf_dir=self.leaf_dir,
                                need_input=True,
                                subject_string=subject_string,
                                model=None,
                                debug=self.debug,
                                train=self.train,
                            )

    def setup_logger(self):
        monitor_dict = self.setup_monitors()
        self.logger = Logger(self.exp_dir, monitor_dict, train=self.train)


    def record_settings(self, passed_args):
        with open(os.path.join(self.root_dir, self.intermediate_dir, 'command.txt'), 'w') as f:
            #import ipdb; ipdb.set_trace()
            out_strs = self.cmd_line_args.split(' -')
            for s in out_strs[1:]:
                f.write('-{}\n'.format(s))

        for k, v in sorted(passed_args.items()): logging.info('\t{} = {}'.format(k, v))

        logging.info("\n")
        logging.info("Git commit = {}".format(get_git_revision_short_hash()))
        logging.info("\tPID = {}".format(self.pid))
        logging.info("\t{}unning on GPU".format("R" if torch.cuda.is_available() else "Not r"))

    def log(self, **kwargs):
        if self.train:
            self.log_train(**kwargs)
        else:
            self.log_test(**kwargs)

    def log_train(self, **kwargs):

        self.logger.log(**kwargs)
        if kwargs['epoch'] == 1 and self.emailer is not None:
            self.emailer.send_msg(self.logger.monitor_collection.monitors['eta'].value, "Job {}-{} on {} ETA: {}".format(self.slurm_array_job_id, self.slurm_array_task_id, self.host.split('.')[0], self.logger.monitor_collection.monitors['eta'].value))
        out_str = "ITERATION {:5}\n".format(
                self.logger.monitor_collection.monitors['iteration'].value)
        out_str += self.logger.monitor_collection.string

        self.signal_handler.results_strings.append(out_str)
        logging.info(out_str)

    def log_test(self,**kwargs):
        self.logger.log(**kwargs)
        out_str = "{:5}\t".format(
                self.logger.monitor_collection.monitors['model'].value)
        for monitor in self.metric_monitors:
            out_str += monitor.string
        self.signal_handler.results_strings.append(out_str)
        logging.info(out_str)

    def save(self, model, settings):
        self.saver.save(model, settings)

    def finished(self):
        self.signal_handler.completed()

    def initial_email(self):
        text = ['JOB STARTED', self.exp_dir, self.host.split('.')[0], str(self.pid)]
        if self.emailer is not None:
            self.emailer.send_msg('\n'.join(text), ' | '.join(text))
