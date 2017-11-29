import shutil
import os
import logging
from misc.constants import FINISHED_MODELS_DIR, DEBUG_MODELS_DIR, KILLED_MODELS_DIR, INTERRUPTED_MODELS_DIR

class Mover:
    def __init__(self, root_dir, leaf_dir):
        self.root_dir = root_dir
        self.leaf_dir = leaf_dir

    def move_to_folder(self, folder):
        src = os.path.join(self.root_dir, self.leaf_dir)
        dst = os.path.join(folder, self.leaf_dir)
        shutil.move(src, dst)
        logging.info('Moved model directory to {}'.format(dst))

    def move_to_heaven(self):
        self.move_to_folder(FINISHED_MODELS_DIR)

    def move_to_morgue(self):
        self.move_to_folder(DEBUG_MODELS_DIR)

    def move_to_hell(self):
        self.move_to_folder(KILLED_MODELS_DIR)

    def move_to_limbo(self):
        self.move_to_folder(INTERRUPTED_MODELS_DIR)
