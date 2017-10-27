import os

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders

import logging
import sys
import datetime
import signal
import shutil
import pickle
import numpy as np
import torch
import resource
import time

from monitors import *
from loggers import StatsLogger

def get_logfile(exp_dir, silent, verbose):
    logfile = os.path.join(exp_dir, 'log.txt')
    logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+",
                        format="%(asctime)-15s %(message)s")
    if not silent:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        if verbose:
            ch.setLevel(logging.INFO)
        else:
            ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        ch.setFormatter(formatter)
        root.addHandler(ch)
    return logfile

class Emailer:
    def __init__(self, sender, password, recipient):
        self.sender = sender
        self.password = password
        self.recipient = recipient

    def send_msg(self, text, subject, attachments=None):
          msg = MIMEMultipart()
          msg['From'] = self.sender
          msg['To'] = self.recipient if self.recipient is not None else self.sender
          msg['Date'] = formatdate(localtime = True)
          msg['Subject'] = subject

          msg.attach(MIMEText(text))

          if attachments is not None:
              for f in attachments:
                  part = MIMEBase('application', "octet-stream")
                  part.set_payload( open(f,"rb").read() )
                  encoders.encode_base64(part)
                  part.add_header('Content-Disposition', 'attachment; filename="{0}"'.format(os.path.basename(f)))
                  msg.attach(part)

          server = smtplib.SMTP('smtp.gmail.com:587')
          server.ehlo()
          server.starttls()
          server.login(self.sender, self.password)
          server.sendmail(self.sender, self.recipient, msg.as_string())
          server.close()
          logging.info("SENT EMAIL")

def timestring():
    dt = datetime.datetime.now()
    d = "{}-{} at {:02d}:{:02d}:{:02d}".format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
    return d

class SignalHandler:
    def __init__(self, emailer=None, exp_dir=None, logfile=None, need_input=False, subject_string="", model=None):
        self.emailer = emailer
        self.results_strings = ["FAILURE: No results to print!"]
        self.need_input = need_input
        self.model = model
        self.logfile = logfile
        self.exp_dir = exp_dir
        self.subject_string = subject_string
        self.done = False
        signal.signal(signal.SIGTERM, self.killed)
        signal.signal(signal.SIGINT, self.interrupted)

    def set_model(self, model):
        self.model = model

    def cleanup(self):
        pass
        #answer = None if self.need_input else "y"
        #while answer not in ["", "y", "Y", "n", "N"]:
        #    answer = input('Cleanup? (Y/n)\n')
        #if answer in ["", "y", "Y"]:
        #    logging.info("Deleting {}".format(self.exp_dir))
        #    os.system("rm {}/*".format(self.exp_dir))
        #    os.system("rm -r {}".format(self.exp_dir))

    def signal_handler(self, signal, cleanup=True):
        d = timestring()
        alert = '{} on {}'.format(signal, timestring())
        logging.warning(alert)
        subject = "Job {} {}".format(signal, self.subject_string)
        text = "{}\n{}\n{}".format(alert, self.results_strings[-1], self.model)
        attachments = [self.logfile]
        self.emailer.send_msg(text, subject, attachments)
        if cleanup:
            self.cleanup()

    def killed(self, signal, frame):
        self.signal_handler(signal='KILLED')
        sys.exit(0)

    def interrupted(self, signal, frame):
        self.signal_handler(signal='INTERRUPTED')
        sys.exit(0)

    def completed(self):
        self.done = True
        self.signal_handler(signal='COMPLETED', cleanup=False)
        sys.exit(0)

    def crashed(self):
        self.signal_handler(signal='CRASHED')
        sys.exit(0)

class ExperimentHandler:
    def __init__(self, args, root_exp_dir):
        self.pid = os.getpid()

        ''' CUDA AND RANDOM SEED '''
        '''----------------------------------------------------------------------- '''
        np.random.seed(args.seed)
        if args.gpu != "" and torch.cuda.is_available():
            torch.cuda.device(args.gpu)
            torch.cuda.manual_seed(args.seed)
        else:
            torch.manual_seed(args.seed)
            pass

        ''' CREATE MODEL DIRECTORY '''
        '''----------------------------------------------------------------------- '''
        #
        dt = datetime.datetime.now()
        filename_exp = '{}-{}/{:02d}-{:02d}-{:02d}_{}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second, args.extra_tag)
        if args.debug:
            filename_exp += '-DEBUG'
        self.exp_dir = os.path.join(root_exp_dir, filename_exp)
        os.makedirs(self.exp_dir)

        ''' SET UP LOGGING '''
        '''----------------------------------------------------------------------- '''
        self.logfile = get_logfile(self.exp_dir, args.silent, args.verbose)
        logging.info(self.logfile)
        logging.info(self.exp_dir)

        ''' SIGNAL HANDLER '''
        '''----------------------------------------------------------------------- '''
        self.emailer=Emailer(args.sender, args.password, args.recipient)
        self.signal_handler = SignalHandler(
                                emailer=self.emailer,
                                logfile=self.logfile,
                                exp_dir=self.exp_dir,
                                need_input=True,
                                subject_string='{}(Logfile = {}, PID = {}, GPU = {})'.format("[DEBUGGING] " if args.debug else "", self.logfile, self.pid, args.gpu),
                                model=None
                                )
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

        ''' RECORD SETTINGS '''
        '''----------------------------------------------------------------------- '''
        logging.info("Logfile at {}".format(self.logfile))
        for k, v in sorted(vars(args).items()): logging.warning('\t{} = {}'.format(k, v))

        logging.warning("\tPID = {}".format(self.pid))
        logging.warning("\t{}unning on GPU".format("R" if torch.cuda.is_available() else "Not r"))



    def usage(self):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

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
        logging.info("FINISHED TRAINING")
        logging.info("Results in {}".format(self.exp_dir))
        self.signal_handler.completed()
        self.stats_logger.close()
