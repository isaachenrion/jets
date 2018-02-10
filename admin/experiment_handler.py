import torch
import logging
import os
import datetime
import numpy as np
import socket
import shutil
from .utils import get_logfile
from .emailer import Emailer
from .signal_handler import SignalHandler
from monitors import *
from .logger import StatsLogger
from misc.constants import RUNNING_MODELS_DIR, ALL_MODEL_DIRS


class ExperimentHandler:
    def __init__(self, args):
        #self.debug = args.debug
        self.pid = os.getpid()
        self.cuda_and_random_seed(args)
        self.create_all_model_dirs()
        self.model_directory(args)
        self.setup_logging(args)
        self.setup_signal_handler(args)
        self.setup_stats_logger(args)
        self.record_settings(args)
        self.initial_email()

    def cuda_and_random_seed(self, args):
        ''' CUDA AND RANDOM SEED '''
        '''----------------------------------------------------------------------- '''
        if args.gpu != "" and torch.cuda.is_available():
            torch.cuda.device(args.gpu)

        if args.seed is None:
            args.seed = np.random.randint(0, 2**16 - 1)
        np.random.seed(args.seed)

        if args.gpu != "" and torch.cuda.is_available():
            torch.cuda.device(args.gpu)
            torch.cuda.manual_seed(args.seed)
        else:
            torch.manual_seed(args.seed)

    def model_directory(self, args):
        self.root_dir = RUNNING_MODELS_DIR
        self.model_type_dir = os.path.join(args.dataset, args.jet_transform)
        dt = datetime.datetime.now()
        self.filename_exp = '{}-{}-{:02d}-{:02d}-{:02d}_{}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second, args.slurm_job_id)
        self.leaf_dir = os.path.join(self.model_type_dir, self.filename_exp)
        self.exp_dir = os.path.join(self.root_dir,self.leaf_dir)
        os.makedirs(self.exp_dir)
        self.start_dt = dt

    def create_all_model_dirs(self):
        for model_dir in ALL_MODEL_DIRS:
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

    def setup_logging(self, args):
        ''' SET UP LOGGING '''
        '''----------------------------------------------------------------------- '''
        self.logfile = get_logfile(self.exp_dir, args.silent, args.verbose)
        self.host = socket.gethostname()
        logging.warning("running on {}".format(self.host))
        logging.warning(self.exp_dir)

    def setup_signal_handler(self, args):
        ''' SIGNAL HANDLER '''
        '''----------------------------------------------------------------------- '''

        with open('misc/email_addresses.txt', 'r') as f:
            lines = f.readlines()
            recipient, sender, password = (l.strip() for l in lines)

        if not args.no_email:
            self.emailer = Emailer(sender, password, recipient)
        else:
            self.emailer = None
        self.signal_handler = SignalHandler(
                                emailer=self.emailer,
                                logfile=self.logfile,
                                root_dir=self.root_dir,
                                leaf_dir=self.leaf_dir,
                                need_input=True,
                                subject_string='{} (Machine = {}, Logfile = {}, PID = {}, GPU = {})'.format("[DEBUGGING] " if args.debug else "", self.host, self.logfile, self.pid, args.gpu),
                                model=None,
                                debug=args.debug,
                                train=args.train,
                            )

    def setup_stats_logger(self, args):
        ''' STATS LOGGER '''
        '''----------------------------------------------------------------------- '''
        roc_auc = ROCAUC()
        inv_fpr = InvFPR()
        best_inv_fpr = Best(inv_fpr)
        epoch_counter = Regurgitate('epoch', visualizing=False)
        batch_counter = Regurgitate('iteration', visualizing=False)
        valid_loss = Regurgitate('valid_loss')
        train_loss = Regurgitate('train_loss')
        #prediction_histogram = EachClassHistogram([0,1], 'yy', 'yy_pred', append=False)
        #logtimer = Regurgitate('logtime')
        logtimer=Collect('logtime', fn='last', visualizing=False)
        epoch_timer=Hours()
        cumulative_timer=Collect('time', fn='sum', visualizing=False)
        eta=ETA(self.start_dt, args.epochs)

        model_file = os.path.join(self.exp_dir, 'model_state_dict.pt')
        settings_file = os.path.join(self.exp_dir, 'settings.pickle')
        self.saver = Saver(best_inv_fpr, model_file, settings_file, visualizing=False)

        monitors = [
            epoch_counter,
            batch_counter,
            roc_auc,
            inv_fpr,
            best_inv_fpr,
            valid_loss,
            train_loss,
            self.saver,
            #prediction_histogram,
            logtimer,
            cumulative_timer,
            epoch_timer,
            eta
        ]

        monitors = {m.name: m for m in monitors}
        self.stats_logger = StatsLogger(self.exp_dir, monitors, args.visualizing, train=True)

    def record_settings(self, args):
        ''' RECORD SETTINGS '''
        '''----------------------------------------------------------------------- '''
        for k, v in sorted(vars(args).items()): logging.warning('\t{} = {}'.format(k, v))

        logging.warning("\tPID = {}".format(self.pid))
        logging.warning("\t{}unning on GPU".format("R" if torch.cuda.is_available() else "Not r"))

    def log(self, **kwargs):
        self.stats_logger.log(**kwargs)
        if kwargs['epoch'] == 1 and self.emailer is not None:
            self.emailer.send_msg(self.stats_logger.monitors['eta'].value, "PID {} on {} ETA: {}".format(self.pid, self.host.split('.')[0], self.stats_logger.monitors['eta'].value))
        if np.isnan(self.stats_logger.monitors['inv_fpr'].value):
            logging.warning("NaN in 1/FPR\n")

        out_str = "{:5d}\t~loss(train)={:.4f}\tloss(valid)={:.4f}\troc_auc(valid)={:.4f}".format(
                kwargs['iteration'],
                kwargs['train_loss'],
                kwargs['valid_loss'],
                self.stats_logger.monitors['roc_auc'].value)

        out_str += "\t1/FPR @ TPR = 0.5: {:.2f}\tBest 1/FPR @ TPR = 0.5: {:.2f}".format(self.stats_logger.monitors['inv_fpr'].value, self.stats_logger.monitors['best_inv_fpr'].value)
        self.signal_handler.results_strings.append(out_str)
        logging.info(out_str)

    def save(self, model, settings):
        self.saver.save(model, settings)

    def finished(self):
        self.stats_logger.complete_logging()
        self.signal_handler.completed()

    def initial_email(self):
        text = ['JOB STARTED', self.exp_dir, self.host.split('.')[0], str(self.pid)]
        if self.emailer is not None:
            self.emailer.send_msg('\n'.join(text), ' | '.join(text))


class EvaluationExperimentHandler(ExperimentHandler):
    def __init__(self, args):
        super().__init__(args)
        self.latex = args.latex

    def model_directory(self, args):
        self.root_dir = args.root_dir
        self.model_type_dir = args.model_dir
        self.leaf_dir = self.model_type_dir
        i = 0
        temp = self.leaf_dir + '/run' + str(i)
        while os.path.exists(os.path.join(self.root_dir,temp)):
            i += 1
            temp = self.leaf_dir + '/run' + str(i)
        self.leaf_dir = temp

        self.exp_dir = os.path.join(self.root_dir,self.leaf_dir)
        print(self.exp_dir)
        os.makedirs(self.exp_dir)

    def setup_stats_logger(self, args):
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
        self.stats_logger = StatsLogger(self.exp_dir, monitors, args.visualizing, train=False)

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
