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

#def summary_email(out_str, model, sigtype):
#    subject = 'JOB {} (Logfile = {}, PID = {}, GPU = {})'.format(sigtype, logfile, pid, args.gpu)
#    subject_string = '(Logfile = {}, PID = {}, GPU = {})'.format(logfile, pid, args.gpu)
#    attachments = [logfile]
#    text = ""
#    text += "{}\n".format(model)
#    text += "{}".format(out_str)
#    send_msg(text, subject, args.sender, args.password, args.recipient, attachments)

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
    def __init__(self, emailer=None, exp_dir=None, logfile=None, need_input=False, subject_string="", monitor_strings=None, model=None):
        self.emailer = emailer
        self.monitor_strings = monitor_strings
        self.need_input = need_input
        self.model = model
        self.logfile = logfile
        self.exp_dir = exp_dir
        self.subject_string = subject_string
        self.no_result_string = "FAILURE: No results to print!"

        signal.signal(signal.SIGTERM, self.signal_term_handler)
        signal.signal(signal.SIGINT, self.signal_int_handler)

    def set_model(self, model):
        self.model = model

    def cleanup(self):
        answer = None if self.need_input else "y"
        while answer not in ["", "y", "Y", "n", "N"]:
            answer = input('Cleanup? (Y/n)\n')
            if answer in ["", "y", "Y"]:
                shutil.rmtree(self.exp_dir)

    def signal_term_handler(self, signal, frame):
        d = timestring()
        alert = 'KILLED on {}'.format(timestring())
        logging.warning(alert)
        subject = "Job {} {}".format("KILLED", self.subject_string)
        if len(self.monitor_strings) > 0:
            text = "{}\n{}\n{}".format(alert, self.monitor_strings[-1], self.model)
        else:
            text = "{}\n{}\n{}".format(alert, self.no_result_string, self.model)

        attachments = [self.logfile]
        self.emailer.send_msg(text, subject, attachments)
        self.cleanup()
        sys.exit(0)

    def signal_int_handler(self, signal, frame):
        d = timestring()
        alert = 'INTERRUPTED on {}'.format(timestring())
        logging.warning(alert)
        subject = "Job {} {}".format("INTERRUPTED", self.subject_string)
        if len(self.monitor_strings) > 0:
            text = "{}\n{}\n{}".format(alert, self.monitor_strings[-1], self.model)
        else:
            text = "{}\n{}\n{}".format(alert, self.no_result_string, self.model)

        attachments = [self.logfile]
        self.emailer.send_msg(text, subject, attachments)
        self.cleanup()
        sys.exit(0)

    def job_completed(self):
        d = timestring()
        alert = 'Completed on {}'.format(timestring())
        logging.warning(alert)
        subject = "Job {} {}".format("Completed", self.subject_string)
        text = "{}\n{}\n{}".format(alert, self.monitor_strings[-1], self.model)
        attachments = [self.logfile]
        self.emailer.send_msg(text, subject, attachments)
        self.cleanup()
        sys.exit(0)
