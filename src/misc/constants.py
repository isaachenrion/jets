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

''' argparse args '''
SRC_DIR = '/Users/isaachenrion/x/research/graphs'
DATA_DIR = os.path.join(SRC_DIR, 'data')
MODELS_DIR = os.path.join(SRC_DIR, 'models')
EMAIL_FILE = os.path.join(SRC_DIR, 'email_addresses.txt')
