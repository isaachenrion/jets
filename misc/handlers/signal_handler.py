import signal
import logging
import sys
from ..utils import timestring
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

    def prepend_to_logfile(self, text):
        with open(self.logfile, 'r') as original: data = original.read()
        with open(self.logfile, 'w') as modified: modified.write("{}\n".format(text) + data)

    def signal_handler(self, signal, cleanup=True):
        d = timestring()
        alert = '{} on {}'.format(signal, timestring())
        logging.warning(alert)
        self.prepend_to_logfile(alert)
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
