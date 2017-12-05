
''' CONSTANTS '''
'''----------------------------------------------------------------------- '''
MODELS_DIR = 'models'
FINISHED_MODELS_DIR = 'finished_models'
DEBUG_MODELS_DIR = 'debug_models'
INTERRUPTED_MODELS_DIR = 'interrupted_models'
KILLED_MODELS_DIR = 'killed_models'
ALL_MODEL_DIRS = [
    MODELS_DIR,
    FINISHED_MODELS_DIR,
    DEBUG_MODELS_DIR,
    INTERRUPTED_MODELS_DIR,
    KILLED_MODELS_DIR,
]

RECIPIENT = "henrion@nyu.edu"
REPORTS_DIR = "reports"
DATASETS = {
    'original':'antikt-kt',
    'pileup':'antikt-kt-pileup25'
}
#RECIPIENT =
''' argparse args '''
STEP_SIZE=0.001
HIDDEN=40
BATCH_SIZE=100
FEATURES=7
DECAY=0.96
EPOCHS=100
ITERS=2
SCALES=-1
SENDER="results74207281@gmail.com"
PASSWORD="deeplearning"
VALID=27000
DATA_DIR = 'data/w-vs-qcd/pickles'
