import torch
import logging
import os
import datetime
import numpy as np
import socket
import shutil

from collections import OrderedDict

from .utils import get_logfile
from .emailer import get_emailer
from .signal_handler import SignalHandler
from .logger import StatsLogger

from ..monitors import *
from ..misc.constants import RUNNING_MODELS_DIR, ALL_MODEL_DIRS


class ExperimentHandler:
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
            **kwargs
            ):

        pid = os.getpid()
        host = socket.gethostname()

        passed_args = locals()
        kwargs = passed_args.pop('kwargs', None)
        passed_args.pop('self')
        passed_args.update(kwargs)

        self.debug = debug
        self.slurm = slurm
        self.slurm_array_task_id = slurm_array_task_id
        self.slurm_array_job_id = slurm_array_job_id
        self.pid = pid
        self.host = host
        self.train = train

        self.cuda_and_random_seed(gpu, seed)
        self.create_all_model_dirs()
        self.setup_model_directory(dataset, model)
        self.setup_logging(silent, verbose)
        self.setup_signal_handler(email)
        self.setup_stats_logger(epochs, visualizing)
        self.record_settings(passed_args)
        self.initial_email()

    def cuda_and_random_seed(self, gpu, seed):
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

    def setup_model_directory(self, dataset, model):
        self.root_dir = RUNNING_MODELS_DIR
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
        for model_dir in ALL_MODEL_DIRS:
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

    def setup_logging(self, silent, verbose):
        ''' SET UP LOGGING '''
        '''----------------------------------------------------------------------- '''
        self.logfile = get_logfile(self.exp_dir, silent, verbose)
        logging.warning("running on {}".format(self.host))
        logging.warning(self.exp_dir)

    def setup_signal_handler(self, email):
        ''' SIGNAL HANDLER '''
        '''----------------------------------------------------------------------- '''

        if email:
            self.emailer = get_emailer()
        else:
            self.emailer = None

        if self.slurm:
            subject_string = '{} (Machine = {}, Logfile = {}, Slurm id = {}-{}, GPU = {})'.format("[DEBUGGING] " if self.debug else "", self.host, self.logfile, self.slurm_array_job_id, self.slurm_array_task_id, self.gpu)
        else:
            subject_string = '{} (Machine = {}, Logfile = {}, PID = {}, GPU = {})'.format("[DEBUGGING] " if self.debug else "", self.host, self.logfile, self.pid, self.gpu)

        self.signal_handler = SignalHandler(
                                emailer=self.emailer,
                                logfile=self.logfile,
                                root_dir=self.root_dir,
                                intermediate_dir=self.intermediate_dir,
                                leaf_dir=self.leaf_dir,
                                need_input=True,
                                subject_string=subject_string,
                                model=None,
                                debug=self.debug,
                                train=self.train,
                            )

    def setup_stats_logger(self, epochs, visualizing):
        ''' STATS LOGGER '''
        '''----------------------------------------------------------------------- '''
        roc_auc = ROCAUC(visualizing=True)
        inv_fpr = InvFPR(visualizing=True)
        best_roc_auc = Best(roc_auc)
        best_inv_fpr = Best(inv_fpr)
        inv_fpr_at_best_roc_auc = LogOnImprovement(inv_fpr, best_roc_auc)
        epoch_counter = Regurgitate('epoch', visualizing=False)
        batch_counter = Regurgitate('iteration', visualizing=False)
        valid_loss = Regurgitate('valid_loss', visualizing=True)
        train_loss = Regurgitate('train_loss', visualizing=True)
        #prediction_histogram = EachClassHistogram([0,1], 'yy', 'yy_pred', append=False)
        #logtimer = Regurgitate('logtime')
        logtimer=Collect('logtime', fn='last', visualizing=False)
        epoch_timer=Hours()
        cumulative_timer=Collect('time', fn='sum', visualizing=False)
        eta=ETA(self.start_dt, epochs)

        model_file = os.path.join(self.exp_dir, 'model_state_dict.pt')
        settings_file = os.path.join(self.exp_dir, 'settings.pickle')
        saver = Saver(best_roc_auc, model_file, settings_file, visualizing=False)

        monitors = [
            epoch_counter,
            batch_counter,
            roc_auc,
            inv_fpr,
            best_roc_auc,
            best_inv_fpr,
            inv_fpr_at_best_roc_auc,
            valid_loss,
            train_loss,
            saver,
            #prediction_histogram,
            logtimer,
            cumulative_timer,
            epoch_timer,
            eta
        ]

        monitor_dict = OrderedDict()
        for m in monitors: monitor_dict[m.name] = m

        self.stats_logger = StatsLogger(self.exp_dir, monitor_dict, visualizing, train=self.train)
        self.saver = saver

    def record_settings(self, passed_args):
        ''' RECORD SETTINGS '''
        '''----------------------------------------------------------------------- '''
        for k, v in sorted(passed_args.items()): logging.warning('\t{} = {}'.format(k, v))

        logging.warning("\tPID = {}".format(self.pid))
        logging.warning("\t{}unning on GPU".format("R" if torch.cuda.is_available() else "Not r"))

    def log(self, **kwargs):
        self.stats_logger.log(**kwargs)
        if kwargs['epoch'] == 1 and self.emailer is not None:
            self.emailer.send_msg(self.stats_logger.monitors['eta'].value, "Job {}-{} on {} ETA: {}".format(self.slurm_array_job_id, self.slurm_array_task_id, self.host.split('.')[0], self.stats_logger.monitors['eta'].value))
        if np.isnan(self.stats_logger.monitors['inv_fpr'].value):
            logging.warning("NaN in 1/FPR\n")

        out_str = "{:5d}\t~loss(train)={:.4f}\tloss(valid)={:.4f}\troc_auc(valid)={:.4f}".format(
                kwargs['iteration'],
                kwargs['train_loss'],
                kwargs['valid_loss'],
                self.stats_logger.monitors['roc_auc'].value)

        out_str += "\t1/FPR @ TPR = 0.5: {:.2f}\tBest roc_auc: {:.5f}".format(self.stats_logger.monitors['inv_fpr'].value, self.stats_logger.monitors['best_roc_auc'].value)
        self.signal_handler.results_strings.append(out_str)
        logging.info(out_str)

    def save(self, model, settings):
        self.saver.save(model, settings)

    def finished(self):
        self.stats_logger.complete_logging()
        self.signal_handler.completed()
        #if not self.slurm:
        #    os.rmdir(os.path.join(self.root_dir, self.intermediate_dir))

    def initial_email(self):
        text = ['JOB STARTED', self.exp_dir, self.host.split('.')[0], str(self.pid)]
        if self.emailer is not None:
            self.emailer.send_msg('\n'.join(text), ' | '.join(text))


class EvaluationExperimentHandler(ExperimentHandler):
    def __init__(self, latex=None, **kwargs):
        super().__init__(**kwargs)
        self.latex = latex

    #def model_directory(self, args):
    #    self.root_dir = args.root_dir
    #    self.model_type_dir = args.model_dir
    #    self.leaf_dir = self.model_type_dir
    #    i = 0
    #    temp = self.leaf_dir + '/run' + str(i)
    #    while os.path.exists(os.path.join(self.root_dir,temp)):
    #        i += 1
    #        temp = self.leaf_dir + '/run' + str(i)
    #    self.leaf_dir = temp
    #    self.exp_dir = os.path.join(self.root_dir,self.leaf_dir)
    #    print(self.exp_dir)
    #    os.makedirs(self.exp_dir)

    def setup_stats_logger(self, _, visualizing):
        ''' STATS LOGGER '''
        '''----------------------------------------------------------------------- '''
        roc_auc = ROCAUC()
        inv_fpr = InvFPR()
        roc_curve = ROCCurve()
        model_counter = Regurgitate('model', visualizing=False)
        logtimer=Collect('logtime', fn='mean')
        prediction_histogram = EachClassHistogram([0,1], 'yy', 'yy_pred', append=True)
        monitors = [
            model_counter,
            roc_auc,
            inv_fpr,
            roc_curve,
            prediction_histogram,
            logtimer
        ]
        monitors = {m.name: m for m in monitors}
        self.stats_logger = StatsLogger(self.exp_dir, monitors, visualizing, train=False)

    def log(self, **kwargs):
        self.stats_logger.log(**kwargs)

        if not self.latex:
            out_str = "\tModel = {}\t1/FPR @ TPR = 0.5={:.4f}\troc_auc={:.4f}".format(
                    kwargs['model'],
                    self.stats_logger.monitors['inv_fpr'].value if kwargs.get('compute_monitors', True) else kwargs['inv_fpr'],
                    self.stats_logger.monitors['roc_auc'].value if kwargs.get('compute_monitors', True) else kwargs['roc_auc']
                    )
        else:
            if not short:
                logging.info("%10s \t& %30s \t& %.4f $\pm$ %.4f \t& %.1f $\pm$ %.1f \\\\" %
                      (input,
                       label,
                       np.mean(rocs),
                       np.std(rocs),
                       np.mean(inv_fprs[:, 225]),
                       np.std(inv_fprs[:, 225])))
            else:
                logging.info("%30s \t& %.4f $\pm$ %.4f \t& %.1f $\pm$ %.1f \\\\" %
                      (label,
                       np.mean(rocs),
                       np.std(rocs),
                       np.mean(inv_fprs[:, 225]),
                       np.std(inv_fprs[:, 225])))

        self.signal_handler.results_strings.append(out_str)
        logging.info(out_str)
