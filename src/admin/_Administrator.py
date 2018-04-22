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
from .emailer import Emailer
from .signal_handler import SignalHandler
from .Logger import Logger

from ..misc.constants import RUNNING_MODELS_DIR, ALL_MODEL_DIRS
from src.monitors import ETA, Saver

def get_git_revision_short_hash():
    s = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    s = str(s).split('\'')[1]
    return s

def cuda_and_random_seed(gpu, seed):
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
    return seed, gpu

def create_all_model_dirs(models_dir):
    for intermediate_dir in ALL_MODEL_DIRS:
        model_dir = os.path.join(models_dir, intermediate_dir)
        if not os.path.isdir(model_dir):
            try:
                os.makedirs(model_dir)
            except FileExistsError:
                pass

def get_experiment_dirname(slurm_array_job_id, pid, train):
    if slurm_array_job_id is not None and train:
        filename_exp = '{}'.format(slurm_array_job_id)
    else:
        dt = datetime.datetime.now()
        filename_exp = '{}-{}-{:02d}-{:02d}-{:02d}_{}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second, pid)
    return filename_exp

def get_experiment_leafname(slurm_array_task_id, train):
    if slurm_array_task_id is not None and train:
        return str(slurm_array_task_id)
    return ''

def setup_model_directory(models_dir, pid, slurm_array_job_id, slurm_array_task_id, train):
    intermediate_dir = get_experiment_dirname(slurm_array_job_id, pid, train)
    leaf_dir = get_experiment_leafname(slurm_array_task_id, train)
    exp_dir = os.path.join(models_dir,intermediate_dir,leaf_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    return exp_dir

def record_cmd_line_args(cmd_line_args, filename):
    with open(filename, 'w') as f:
        out_strs = cmd_line_args.split(' -')
        for s in out_strs[1:]:
            f.write('-{}\n'.format(s))

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
            epochs=None,
            debug=None,
            slurm_array_task_id=None,
            slurm_array_job_id=None,
            gpu=None,
            seed=None,
            email_filename=None,
            silent=None,
            verbose=None,
            cmd_line_args=None,
            models_dir=None,
            monitor_collection=None,
            arg_string=None,
            ):


        pid = os.getpid()
        host = socket.gethostname()
        slurm = slurm_array_job_id is not None

        # Create all of the necessary directories for the experiment
        create_all_model_dirs(models_dir)
        _temp = get_experiment_dirname(slurm_array_job_id, pid, train)
        intermediate_dir = os.path.join(dataset, model, _temp)
        leaf_dir = get_experiment_leafname(slurm_array_task_id, train)
        exp_dir = os.path.join(models_dir,RUNNING_MODELS_DIR,intermediate_dir,leaf_dir)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        logfile = get_logfile(exp_dir, silent, verbose)

        if email_filename is not None and not slurm:
            emailer = Emailer.from_filename(email_filename)
        else:
            emailer = None

        if slurm:
            subject_string = '{} (Machine = {}, Logfile = {}, Slurm id = {}-{}, GPU = {})'.format("[DEBUGGING] " if debug else "", host, logfile, slurm_array_job_id, slurm_array_task_id, gpu)
        else:
            subject_string = '{} (Machine = {}, Logfile = {}, PID = {}, GPU = {})'.format("[DEBUGGING] " if debug else "", host, logfile, pid, gpu)

        signal_handler = SignalHandler(
                                emailer=emailer,
                                logfile=logfile,
                                current_dir=RUNNING_MODELS_DIR,
                                models_dir=models_dir,
                                intermediate_dir=intermediate_dir,
                                leaf_dir=leaf_dir,
                                need_input=True,
                                subject_string=subject_string,
                                model=None,
                                debug=debug,
                                train=train,
                            )

        eta = ETA(datetime.datetime.now(), epochs)
        model_file = os.path.join(exp_dir, 'model_state_dict.pt')
        settings_file = os.path.join(exp_dir, 'settings.pickle')
        saver = Saver(monitor_collection.track_monitor, model_file, settings_file, visualizing=False, printing=False)
        monitor_collection.add_monitors(saver, eta, initialize=False)

        logger = Logger(exp_dir, monitor_collection, train=train)

        cmd_file = os.path.join(models_dir, RUNNING_MODELS_DIR, intermediate_dir, 'command.txt')
        record_cmd_line_args(cmd_line_args, cmd_file)

        seed, gpu = cuda_and_random_seed(gpu, seed)

        logging.info("running on {}".format(host))
        logging.info(exp_dir)
        logging.info(arg_string)
        logging.info("\n")
        logging.info("Git commit = {}".format(get_git_revision_short_hash()))
        logging.info("\tPID = {}".format(pid))
        logging.info("\t{}unning on GPU".format("R" if torch.cuda.is_available() else "Not r"))
        logging.info("Seed = {}".format(seed))
        logging.info("GPU = {}".format(gpu))

        msg = ['JOB STARTED', exp_dir, host.split('.')[0], str(pid)]
        text = '\n'.join(msg)
        subject = ' | '.join(msg)
        try:
            emailer.send_msg(text, subject)
        except AttributeError:
            logging.info(subject)


        #### set up private attributes
        self._signal_handler = signal_handler
        self._emailer = emailer
        self._train = train
        self._saver = saver

        #### set up public attributes
        self.logger = logger



    def set_model(self, model):
        self._signal_handler.set_model(model)

    def log(self, **kwargs):
        if self._train:
            self._log_train(**kwargs)
        else:
            self._log_test(**kwargs)

    def _log_train(self, **kwargs):

        self.logger.log(**kwargs)
        out_str = "ITERATION {:5}\n".format(
                self.logger.monitor_collection.monitors['iteration'].value)
        out_str += self.logger.monitor_collection.string

        self._signal_handler.results_strings.append(out_str)
        logging.info(out_str)

    def _log_test(self,**kwargs):
        self.logger.log(**kwargs)
        out_str = "{:5}\t".format(
                self.logger.monitor_collection.monitors['model'].value)
        for monitor in self.metric_monitors:
            out_str += monitor.string
        self._signal_handler.results_strings.append(out_str)
        logging.info(out_str)

    def save(self, model, settings):
        self._saver.save(model, settings)

    def finished(self):
        self._signal_handler.completed()
