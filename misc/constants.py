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

with open('misc/email_addresses.txt', 'r') as f:
    lines = f.readlines()
    RECIPIENT, SENDER, PASSWORD = (l.strip() for l in lines)

REPORTS_DIR = "reports"
DATASETS = {
    'original':'antikt-kt',
    'pileup':'antikt-kt-pileup25'
}

''' argparse args '''
STEP_SIZE=0.001
HIDDEN=40
BATCH_SIZE=100
FEATURES=7
DECAY=0.96
EPOCHS=50
ITERS=2
SCALES=-1
VALID=27000
DATA_DIR = 'data/w-vs-qcd/pickles'
