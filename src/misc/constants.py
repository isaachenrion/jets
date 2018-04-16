import os

''' CONSTANTS '''
'''----------------------------------------------------------------------- '''

RUNNING_MODELS_DIR = 'running'
FINISHED_MODELS_DIR = 'finished'
DEBUG_MODELS_DIR = 'debugging'
INTERRUPTED_MODELS_DIR = 'interrupted'
KILLED_MODELS_DIR = 'killed'
ARCHIVED_MODELS_DIR = 'archive'
ALL_MODEL_DIRS = [
    RUNNING_MODELS_DIR,
    FINISHED_MODELS_DIR,
    DEBUG_MODELS_DIR,
    INTERRUPTED_MODELS_DIR,
    KILLED_MODELS_DIR,
    ARCHIVED_MODELS_DIR,
]

REPORTS_DIR = "reports"

w_vs_qcd = 'w-vs-qcd'
quark_gluon = 'quark-gluon'
DATASETS = {
    'w':(w_vs_qcd,'antikt-kt'),
    'wp':(w_vs_qcd + '/pileup','antikt-kt-pileup25-new'),
    'pp': (quark_gluon,'pp'),
    'pbpb': (quark_gluon,'pbpb'),
    'protein': ('proteins', 'casp11')
}

''' argparse args '''
SRC_DIR = '/Users/isaachenrion/x/research/graphs'
DATA_DIR = os.path.join(SRC_DIR, 'data')
MODELS_DIR = os.path.join(SRC_DIR, 'models')
EMAIL_FILE = os.path.join(SRC_DIR, 'email_addresses.txt')
