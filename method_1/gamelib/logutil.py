# Simple logging util.

import logging
import pathlib


output_root = '../../gamelife_data/method_1/'
kaggle_root = '../../gamelife_data/kaggle/'

def init_game_log(title, lvl = logging.INFO, top_dir = None):
    # top_dir is the root of the program, outside of githut local repo
    global output_root
    if top_dir:
        output_root = top_dir
    pathlib.Path(output_root + 'nn_predict').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename = output_root + 'gamelife.log', 
        level = lvl, 
        format = '%(asctime)s | %(levelname)s | %(message)s')
    logging.info('Game of life [' + title + '] started logging.')

