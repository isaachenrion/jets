import shutil
import os
import logging
from misc.constants import FINISHED_MODELS_DIR, DEBUG_MODELS_DIR, KILLED_MODELS_DIR, INTERRUPTED_MODELS_DIR

class Mover:
    def __init__(self, root_dir, intermediate_dir, leaf_dir):
        self.current_root = root_dir
        self.leaf_dir = leaf_dir
        self.intermediate_dir = intermediate_dir

    def move_to_folder(self, folder):
        src = os.path.join(self.current_root, self.intermediate_dir, self.leaf_dir)
        dst = os.path.join(folder, self.intermediate_dir, self.leaf_dir)
        intermediate_path = os.path.join(folder, self.intermediate_dir)
        if not os.path.exists(os.makedirs(intermediate_path)):
            os.makedirs(intermediate_path)
        shutil.move(src, dst)
        logging.info('Moved model directory to {}'.format(dst))
        self.current_root = folder

    def move_to_finished(self):
        self.move_to_folder(FINISHED_MODELS_DIR)

    def move_to_debug(self):
        self.move_to_folder(DEBUG_MODELS_DIR)

    def move_to_killed(self):
        self.move_to_folder(KILLED_MODELS_DIR)

    def move_to_interrupted(self):
        self.move_to_folder(INTERRUPTED_MODELS_DIR)
