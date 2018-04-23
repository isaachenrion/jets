import signal
import logging
import sys

from .utils import timestring
from .mover import Mover

class SignalHandler:
    def __init__(self,
            emailer=None,
            root_dir=None,
            current_dir=None,
            leaf_dir=None,
            intermediate_dir=None,
            logfile=None,
            need_input=False,
            subject_string="",
            model=None,
            debug=None,
            train=None,
        ):
        self.emailer = emailer
        self.mover = Mover(root_dir, current_dir, intermediate_dir, leaf_dir)
        self.results_strings = ["FAILURE: No results to print!"]
        self.need_input = need_input
        self.model = model
        self.logfile = logfile
        self.subject_string = subject_string
        self.done = False
        self.train = train
        self.debug = debug
        signal.signal(signal.SIGTERM, self.killed)
        signal.signal(signal.SIGINT, self.interrupted)

    def set_model(self, model):
        self.model = model

    def prepend_to_logfile(self, text):
        with open(self.logfile, 'r') as original: data = original.read()
        with open(self.logfile, 'w') as modified: modified.write("{}\n".format(text) + data)

    def signal_admin(self, signal, cleanup=True):
        d = timestring()
        alert = '{} on {}'.format(signal, timestring())
        logging.info(alert)
        self.prepend_to_logfile(alert)
        subject = "Job {} {}".format(signal, self.subject_string)
        text = "{}\n{}\n{}".format(alert, self.results_strings[-1], self.model)
        attachments = [self.logfile]
        if self.emailer is not None:
            self.emailer.send_msg(text, subject, attachments)

    def killed(self, signal, frame):
        self.signal_admin(signal='KILLED')
        if self.train: self.mover.move_to_killed()
        sys.exit(0)

    def interrupted(self, signal, frame):
        self.signal_admin(signal='INTERRUPTED')
        if self.train: self.mover.move_to_interrupted()
        sys.exit(0)

    def completed(self):
        self.done = True
        self.signal_admin(signal='COMPLETED')
        if self.train:
            if self.debug:
                self.mover.move_to_debug()
            else:
                self.mover.move_to_finished()


    def crashed(self):
        self.signal_admin(signal='CRASHED')
