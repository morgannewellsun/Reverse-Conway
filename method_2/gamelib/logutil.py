# Simple logging util.

import logging
import pathlib
import time


output_root = '../../gamelife_data/method_2/'
kaggle_root = '../../gamelife_data/kaggle/'

def init_game_log(title, lvl = logging.INFO, top_dir = None):
    # top_dir is the root of the program, outside of githut local repo
    global output_root
    if top_dir:
        output_root = top_dir
    pathlib.Path(output_root + 'nn_predict').mkdir(parents=True, exist_ok=True)
    log_path = output_root + 'gamelife.log'
    logging.basicConfig(
        filename = log_path, 
        level = lvl, 
        format = '%(asctime)s | %(levelname)s | %(message)s')
    msg = 'Game of life [{}] started logging at {}.'.format(title, log_path)
    logging.critical(msg)
    print(msg)


prev_t = time.time()

def timing():
    global prev_t
    t = time.time()
    t_sec = round(t - prev_t)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    prev_t = t
    return '{}:{}:{}'.format(t_hour, t_min, t_sec)
