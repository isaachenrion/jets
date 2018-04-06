import os
import logging
import sys
import datetime


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_logfile(exp_dir, silent, verbose):
    logfile = os.path.join(exp_dir, 'log.txt')

    logging.basicConfig(level=logging.INFO, filename=logfile, filemode="a+",
                        format="%(message)s")

    debugfile = os.path.join(exp_dir, 'debug.txt')
    ch_debug = logging.StreamHandler(debugfile)
    ch_debug.setLevel(logging.DEBUG)

    if not silent:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        if verbose:
            ch.setLevel(logging.INFO)
        else:
            ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        root.addHandler(ch)
    return logfile

def timestring():
    dt = datetime.datetime.now()
    d = "{}-{} at {:02d}:{:02d}:{:02d}".format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
    return d
