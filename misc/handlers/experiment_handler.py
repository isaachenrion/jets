import torch
import logging
import os
import datetime
import numpy as np
import socket
from ..utils import *
from .emailer import Emailer
from .signal_handler import SignalHandler
from monitors.monitors import *
from ..loggers import StatsLogger


class ExperimentHandler:
    def __init__(self, args):
        self.pid = os.getpid()
        self.cuda_and_random_seed(args)
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
        ''' CREATE MODEL DIRECTORY '''
        '''----------------------------------------------------------------------- '''
        #
        dt = datetime.datetime.now()
        filename_exp = '{}-{}-{:02d}-{:02d}-{:02d}_{}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second, args.extra_tag)
        if args.debug:
            filename_exp += '-DEBUG'
        self.exp_dir = os.path.join(args.root_exp_dir, filename_exp)
        os.makedirs(self.exp_dir)

    def setup_logging(self, args):
        ''' SET UP LOGGING '''
        '''----------------------------------------------------------------------- '''
        self.logfile = get_logfile(self.exp_dir, args.silent, args.verbose)
        self.host = socket.gethostname()
        logging.info("running on {}".format(self.host))
        logging.info(self.logfile)
        logging.info(self.exp_dir)

    def setup_signal_handler(self, args):
        ''' SIGNAL HANDLER '''
        '''----------------------------------------------------------------------- '''
        self.emailer=Emailer(args.sender, args.password, args.recipient)
        self.signal_handler = SignalHandler(
                                emailer=self.emailer,
                                logfile=self.logfile,
                                exp_dir=self.exp_dir,
                                need_input=True,
                                subject_string='{} (Machine = {}, Logfile = {}, PID = {}, GPU = {})'.format("[DEBUGGING] " if args.debug else "", self.host, self.logfile, self.pid, args.gpu),
                                model=None
                            )
    def setup_stats_logger(self, args):
        ''' STATS LOGGER '''
        '''----------------------------------------------------------------------- '''
        roc_auc = ROCAUC()
        inv_fpr = InvFPR()
        best_inv_fpr = Best(inv_fpr)
        epoch_counter = Regurgitate('epoch')
        batch_counter = Regurgitate('iteration')
        valid_loss = Regurgitate('valid_loss')
        train_loss = Regurgitate('train_loss')
        model_file = os.path.join(self.exp_dir, 'model_state_dict.pt')
        settings_file = os.path.join(self.exp_dir, 'settings.pickle')
        self.saver = Saver(best_inv_fpr, model_file, settings_file)
        self.monitors = [
            epoch_counter,
            batch_counter,
            roc_auc,
            inv_fpr,
            best_inv_fpr,
            valid_loss,
            train_loss,
            self.saver,
        ]
        self.statsfile = os.path.join(self.exp_dir, 'stats')
        self.stats_logger = StatsLogger(self.statsfile, headers=[m.name for m in self.monitors])
        self.monitors = {m.name: m for m in self.monitors}

    def record_settings(self, args):
        ''' RECORD SETTINGS '''
        '''----------------------------------------------------------------------- '''
        logging.info("Logfile at {}".format(self.logfile))
        for k, v in sorted(vars(args).items()): logging.warning('\t{} = {}'.format(k, v))

        logging.warning("\tPID = {}".format(self.pid))
        logging.warning("\t{}unning on GPU".format("R" if torch.cuda.is_available() else "Not r"))

    def log(self, **kwargs):
        stats_dict = {}
        for name, monitor in self.monitors.items():
            stats_dict[name] = monitor(**kwargs)
        self.stats_logger.log(stats_dict)

        if np.isnan(self.monitors['inv_fpr'].value):
            logging.warning("NaN in 1/FPR\n")

        out_str = "{:5d}\t~loss(train)={:.4f}\tloss(valid)={:.4f}\troc_auc(valid)={:.4f}".format(
                kwargs['iteration'],
                kwargs['train_loss'],
                kwargs['valid_loss'],
                self.monitors['roc_auc'].value)

        out_str += "\t1/FPR @ TPR = 0.5: {:.2f}\tBest 1/FPR @ TPR = 0.5: {:.2f}".format(self.monitors['inv_fpr'].value, self.monitors['best_inv_fpr'].value)
        self.signal_handler.results_strings.append(out_str)
        logging.info(out_str)

    def save(self, model, settings):
        self.saver.save(model, settings)

    def finished(self):
        finished_training = "FINISHED TRAINING"
        logging.info(finished_training)
        logging.info("Results in {}".format(self.exp_dir))
        self.signal_handler.completed()
        self.stats_logger.close()

    def initial_email(self):
        text = ['JOB STARTED', self.exp_dir, self.host.split('.')[0], str(self.pid)]
        self.emailer.send_msg('\n'.join(text), ' | '.join(text))


class EvaluationExperimentHandler(ExperimentHandler):
    def __init__(self, args):
        super().__init__(args)
        self.latex = args.latex

    def model_directory(self, args):
        ''' CREATE MODEL DIRECTORY '''
        '''----------------------------------------------------------------------- '''
        #
        #dt = datetime.datetime.now()
        filename_exp = '{}'.format(args.root_model_dir)
        if args.debug:
            filename_exp += '-DEBUG'
        self.exp_dir = os.path.join(args.root_exp_dir, filename_exp)
        i = 0
        temp = self.exp_dir + '/run' + str(i)
        while os.path.exists(temp):
            i += 1
            temp = self.exp_dir + '/run' + str(i)
        self.exp_dir = temp
        os.makedirs(self.exp_dir)

    def setup_stats_logger(self, args):
        ''' STATS LOGGER '''
        '''----------------------------------------------------------------------- '''
        roc_auc = ROCAUC()
        inv_fpr = InvFPR()
        roc_curve = ROCCurve()
        model_counter = Regurgitate('model')
        self.monitors = [
            model_counter,
            roc_auc,
            inv_fpr,
            roc_curve,
        ]
        self.statsfile = os.path.join(self.exp_dir, 'stats')
        self.stats_logger = StatsLogger(self.statsfile, headers=[m.name for m in self.monitors if m.scalar])
        self.monitors = {m.name: m for m in self.monitors}

    def log(self, **kwargs):
        stats_dict = {}
        for name, monitor in self.monitors.items():
            monitor_value = monitor(**kwargs)
            if monitor.scalar:
                stats_dict[name] = monitor_value
        self.stats_logger.log(stats_dict)

        if not self.latex:
            out_str = "\tModel = {}\t1/FPR @ TPR = 0.5={:.4f}\troc_auc={:.4f}".format(
                    kwargs['model'],
                    self.monitors['inv_fpr'].value,
                    self.monitors['roc_auc'].value
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

        #out_str += "\t1/FPR @ TPR = 0.5: {:.2f}\tBest 1/FPR @ TPR = 0.5: {:.2f}".format(self.monitors['inv_fpr'].value, self.monitors['best_inv_fpr'].value)
        self.signal_handler.results_strings.append(out_str)
        logging.info(out_str)
