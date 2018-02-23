import os

''' CONSTANTS '''
'''----------------------------------------------------------------------- '''
MODELS_DIR = 'models'
RUNNING_MODELS_DIR = os.path.join(MODELS_DIR,'running')
FINISHED_MODELS_DIR = os.path.join(MODELS_DIR,'finished')
DEBUG_MODELS_DIR = os.path.join(MODELS_DIR,'debugging')
INTERRUPTED_MODELS_DIR = os.path.join(MODELS_DIR,'interrupted')
KILLED_MODELS_DIR = os.path.join(MODELS_DIR,'killed')
ARCHIVED_MODELS_DIR = os.path.join(MODELS_DIR,'archive')
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
    'wp':(w_vs_qcd,'antikt-kt-pileup25-new'),
    'pp': (quark_gluon,'pp'),
    'pbpb': (quark_gluon,'pbpb'),
    #'quark_pp':(quark_gluon,'quark_pp'),
    #'quark_pbpb':(quark_gluon,'quark_pbpb'),
    #'gluon_pbpb':(quark_gluon,'gluon_pbpb'),
    #'gluon_pp':(quark_gluon,'gluon_pp')
}

''' argparse args '''
STEP_SIZE=0.0003
DECAY=1
L2_REGULARIZATION=0.0

FEATURES=7
HIDDEN=64

BATCH_SIZE=100
EPOCHS=100

VALID=27000
DATA_DIR = 'data'
