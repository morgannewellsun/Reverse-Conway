
# Attempt to solve the reverse game by genetic algorithm.

import numpy as np
import pandas as pd
import time
import logging
from gamelib.reversega import ReverseGa
from gamelib.logutil import init_game_log, kaggle_root


init_game_log('Reverse solver by neural net')

# Max number of rows from the train/test.csv files.
# Kaggle supplied training file has 50,001 lines
# Use small numbers to test first.
max_csv_rows = 200

def timing(title):
    global prev_t
    t = time.time()
    t_sec = round(t - prev_t)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    prev_t = t
    print('{} used {}:{}:{}'.format(title, t_hour, t_min, t_sec))


prev_t = time.time()
board = (25, 25)
board_size = np.product(board)
err_stats = [0] * board_size
ga = ReverseGa(nrows=25, ncols=25, pop_size=100, max_iters=400,
               crossover_rate=1, mutation_rate=0.3, tracking=True)

data = pd.read_csv(kaggle_root + 'test.csv', index_col=0, dtype='int', nrows=max_csv_rows)
for idx, row in data.iterrows():
    state = row[1:].tolist()
    err_cnt = ga.revert(row[0], state, [state])
    err_stats[err_cnt] += 1
    msg = 'Run {} has error {}'.format(idx, err_cnt)
    print(msg)
    logging.info(msg)

accuracy = 1 - sum(err_stats) / len(data) / board_size
err_stats = pd.Series(err_stats)
err_stats = err_stats[err_stats > 0]
logging.info('Error Counts:\n{}'.format(err_stats))
logging.info('Accuracy = {}'.format(accuracy))
print('All are done.')
