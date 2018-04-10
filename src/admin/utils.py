import os
import logging
import sys
import datetime
import gc
from functools import reduce
import operator as op
import torch
if torch.cuda.is_available():
    import GPUtil

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


def log_gpu_usage():
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        gpu_util = float(gpus[0].memoryUsed)
        gpu_total = float(gpus[0].memoryTotal)
        logging.info("GPU UTIL: {}/{}. {:.1f}% used".format(gpu_util, gpu_total, 100*gpu_util/gpu_total))
    else:
        pass

def see_tensors_in_memory(ndim=0):
    tensor_list = []
    for obj in gc.get_objects():
        try:
            cond = torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))
            if cond and len(obj.size()) == ndim:
                tensor_list.append((4*reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size()))
        except Exception:
            pass
    total_bytes = reduce(lambda x,y:x+y, map(lambda x:x[0], tensor_list)) if len(tensor_list) > 0 else 0
    logging.info('There are {} tensors in memory, consuming {} total'.format(len(tensor_list), get_bytes(total_bytes)))
    
def clear_all_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except Exception:
            pass

def get_bytes(n):
    n = str(n)
    if len(n) < 4:
        return "{} b".format(n)
    elif len(n) < 7:
        return "{} kb".format(n[:-3])
    elif len(n) < 10:
        return "{} Mb".format(n[:-6])
    elif len(n) < 14:
        return "{} Gb".format(n[:-9])
    else:
        return "{} Tb".format(n[:-12])
