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

from .utils import get_logfile, timestring
from .emailer import Emailer
from .signal_handler import SignalHandler
from .Logger import Logger

from ..misc.constants import RUNNING_MODELS_DIR, ALL_MODEL_DIRS
from src.monitors import ETA, Saver

def get_git_revision_short_hash():
    s = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    s = str(s).split('\'')[1]
    return s

def random_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2**16 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def create_all_model_dirs(root_dir):
    for intermediate_dir in ALL_MODEL_DIRS:
        model_dir = os.path.join(root_dir, intermediate_dir)
        if not os.path.isdir(model_dir):
            try:
                os.makedirs(model_dir)
            except FileExistsError:
                pass

def get_experiment_dirname(slurm_array_job_id, train):
    if slurm_array_job_id is not None and train:
        filename_exp = '{}'.format(slurm_array_job_id)
    else:
        dt = datetime.datetime.now()
        filename_exp = '{}-{}-{:02d}-{:02d}-{:02d}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
    return filename_exp

def get_experiment_leafname(slurm_array_task_id, train):
    if slurm_array_task_id is not None and train:
        return str(slurm_array_task_id)
    return ''

def setup_model_directory(root_dir, slurm_array_job_id, slurm_array_task_id, train):
    intermediate_dir = get_experiment_dirname(slurm_array_job_id, train)
    leaf_dir = get_experiment_leafname(slurm_array_task_id, train)
    exp_dir = os.path.join(root_dir,intermediate_dir,leaf_dir)
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
            slurm_array_task_id=None,
            slurm_array_job_id=None,
            monitor_collections=None,
            root_dir=None,
            intermediate_dir=None,
            leaf_dir=None,
            exp_dir=None,
            current_dir=None,
            silent=None,
            verbose=None,
            email_filename=None,
            debug=None,
            gpu=None,
            cmd_line_args=None,
            seed=None,
            arg_string=None,
            saver=None,
            ):
        slurm = slurm_array_job_id is not None
        pid = os.getpid()
        host = socket.gethostname()

        # Create all of the necessary directories for the experiment

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
                                current_dir=current_dir,
                                root_dir=root_dir,
                                intermediate_dir=intermediate_dir,
                                leaf_dir=leaf_dir,
                                need_input=True,
                                subject_string=subject_string,
                                model=None,
                                debug=debug,
                                train=train,
                            )


        logger = Logger(exp_dir, train, monitor_collections)

        cmd_file = os.path.join(root_dir, current_dir, intermediate_dir, 'command.txt')
        record_cmd_line_args(cmd_line_args, cmd_file)

        seed = random_seed(seed)
        use_cuda = (gpu is not None) and torch.cuda.is_available()
        print(use_cuda)
        if use_cuda:
            torch.cuda.device(gpu)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)

        logging.info("Experiment started: {}".format(timestring()))
        logging.info("running on {}".format(host))
        logging.info(exp_dir)
        logging.info(arg_string)
        logging.info("\n")
        logging.info("Git commit = {}".format(get_git_revision_short_hash()))
        logging.info("\tPID = {}".format(pid))
        logging.info("\t{}unning on GPU".format("R" if use_cuda else "Not r"))
        logging.info("Seed = {}".format(seed))
        logging.info("GPU = {}".format(gpu if use_cuda else "None"))

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
        #self._monitor_collection = monitor_collection

        #### set up public attributes
        self.logger = logger

    def set_model(self, model):
        self._signal_handler.set_model(model)

    def log(self, **kwargs):
        out_str = self.logger.log(**kwargs)
        self._signal_handler.results_strings.append(out_str)
        logging.info(out_str)

    def save(self, model, settings):
        self._saver.save(model, settings)

    def finished(self):
        self._signal_handler.completed()

    @classmethod
    def train(cls,
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
        root_dir=None,
        monitor_collections=None,
        arg_string=None
        ):

        slurm = slurm_array_job_id is not None

        current_dir=RUNNING_MODELS_DIR
        create_all_model_dirs(root_dir)
        _temp = get_experiment_dirname(slurm_array_job_id, train=True)
        intermediate_dir = os.path.join(dataset, model, _temp)
        leaf_dir = get_experiment_leafname(slurm_array_task_id, train=True)
        exp_dir = os.path.join(root_dir,current_dir,intermediate_dir,leaf_dir)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        valid_monitor_collection = monitor_collections['valid']
        model_file = os.path.join(exp_dir, 'model_state_dict.pt')
        settings_file = os.path.join(exp_dir, 'settings.pickle')
        saver = Saver(valid_monitor_collection.track_monitor, model_file, settings_file, visualizing=False, printing=False)
        valid_monitor_collection.add_monitors(saver, initialize=False)
        monitor_collections['valid'] = valid_monitor_collection

        dirs = dict(
            root_dir=root_dir,
            intermediate_dir=intermediate_dir,
            leaf_dir=leaf_dir,
            exp_dir=exp_dir,
            current_dir=current_dir
        )

        init_kwargs = dict(
            train=True,
            silent=silent,
            verbose=verbose,
            monitor_collections=monitor_collections,
            slurm_array_job_id=slurm_array_job_id,
            slurm_array_task_id=slurm_array_task_id,
            email_filename=email_filename,
            debug=debug,
            gpu=gpu,
            cmd_line_args=cmd_line_args,
            seed=seed,
            arg_string=arg_string,
            saver=saver,
            **dirs
        )
        return cls(**init_kwargs)

    @classmethod
    def test(cls,
        dataset=None,
        n_models=None,
        debug=None,
        slurm_array_task_id=None,
        slurm_array_job_id=None,
        gpu=None,
        seed=None,
        email_filename=None,
        silent=None,
        verbose=None,
        cmd_line_args=None,
        root_dir=None,
        monitor_collections=None,
        arg_string=None
        ):

        slurm = slurm_array_job_id is not None

        current_dir=RUNNING_MODELS_DIR
        create_all_model_dirs(root_dir)
        _temp = get_experiment_dirname(slurm_array_job_id, train=False)
        intermediate_dir = os.path.join(dataset, _temp)
        leaf_dir = get_experiment_leafname(slurm_array_task_id, train=False)
        exp_dir = os.path.join(root_dir,current_dir,intermediate_dir,leaf_dir)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        eta = ETA(datetime.datetime.now(), n_models)

        dirs = dict(
            root_dir=root_dir,
            intermediate_dir=intermediate_dir,
            leaf_dir=leaf_dir,
            exp_dir=exp_dir,
            current_dir=current_dir
        )

        init_kwargs = dict(
            train=False,
            silent=silent,
            verbose=verbose,
            monitor_collections=monitor_collections,
            slurm_array_job_id=slurm_array_job_id,
            slurm_array_task_id=slurm_array_task_id,
            email_filename=email_filename,
            debug=debug,
            gpu=gpu,
            cmd_line_args=cmd_line_args,
            seed=seed,
            arg_string=arg_string,
            saver=None,
            **dirs
        )
        return cls(**init_kwargs)
