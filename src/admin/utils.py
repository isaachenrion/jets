import os
import logging
import sys
import datetime
import gc
from functools import reduce
import operator as op
import torch
import numpy as np

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
        logging.info("GPU: {}/{}MiB\t{:.1f}% used".format(int(gpu_util), int(gpu_total), 100*gpu_util/gpu_total))
    else:
        pass

def get_tensors_in_memory(ndim=None):
    tensor_list = []
    for obj in gc.get_objects():
        try:
            cond = torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))
            if ndim is not None:
                cond = cond and len(obj.size()) == ndim
            if cond:
                tensor_list.append(obj)
        except Exception:
            pass
    return tensor_list

def memory_footprint(x):
    b = np.prod(x.size()) * 4
    if isinstance(x, torch.autograd.Variable) and x.grad is not None:
        b = b * 2
    return b

def get_bytes(tensor_list):
    total_bytes = 0
    for t in tensor_list:
        b = memory_footprint(t)
        total_bytes += b
    return total_bytes

def see_tensors_in_memory(ndim=None, summary=False, cuda=False):
    tensor_list = get_tensors_in_memory(ndim)
    tensor_list = (filter(lambda x: not isinstance(x, torch.nn.Parameter), tensor_list))
    if cuda:
        tensor_list = list(filter(lambda x: x.is_cuda, tensor_list))
        cuda_or_cpu = 'CUDA'
    else:
        tensor_list = list(filter(lambda x: not x.is_cuda, tensor_list))
        cuda_or_cpu = 'CPU'

    #if summary:
    total_bytes = get_bytes(tensor_list)

    logging.info((
        'There are {} instantiated {} tensors in memory, consuming {} total.\n'
        'This does not count internal tensors created by pytorch or weights, only things '
        'like variables')
        .format(len(tensor_list), cuda_or_cpu, format_bytes(total_bytes)))

    if not summary:
        logging.info('\n'.join([','.join(list(str(x) for x in t.shape)) for t in tensor_list]))


def compute_model_size(model):
    return sum(memory_footprint(p) for p in model.parameters())

class memory_snapshot:
    def __init__(self, ndim=None, cuda=False, summary=False):
        self.cuda = cuda
        self.ndim=ndim
        self.summary=summary

    def __enter__(self):
        logging.info("BEFORE")
        see_tensors_in_memory(self.ndim, self.summary, self.cuda)

    def __exit__(self, type, value, traceback):
        logging.info("AFTER")
        see_tensors_in_memory(self.ndim, self.summary, self.cuda)


def format_bytes(n):
    n = str(int(n))
    if len(n) < 4:
        return "{} B".format(n)
    elif len(n) < 7:
        return "{} kB".format(n[:-3])
    elif len(n) < 10:
        return "{}.{} MB".format(n[:-6], n[-6:-5])
    elif len(n) < 14:
        return "{}.{} GB".format(n[:-9], n[-9:-8])
    return "{}.{} TB".format(n[:-12], n[-12:-11])
