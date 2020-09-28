
# Attempt to solve the reverse game by genetic algorithm.

import pandas as pd
import logging
from gamelib.reversega import ReverseGa
from gamelib.logutil import init_game_log, kaggle_root
from gamelib.conwaymap import ConwayMap

init_game_log('Reverse solver by genetic algorithm', lvl = logging.CRITICAL)

# Max number of rows from the train/test.csv files.
# Kaggle supplied training file has 50,001 lines
# Use small numbers to test first.
max_csv_rows = 60000

conway = ConwayMap(nrows=25, ncols=25)
ga = ReverseGa(conway, pop_size=10, max_iters=15,
               crossover_rate=1, mutation_rate=0.5, tracking=False)
err_stats = [0] * conway.size

data = pd.read_csv(kaggle_root + 'test.csv', index_col=0, dtype='str', nrows=max_csv_rows)
for idx, row in data.iterrows():
    state = conway.str_to_bin(''.join(row[1:]))
    err_cnt = ga.revert(int(row[0]), state, {state})
    err_stats[err_cnt] += 1
    msg = 'Run {} has error {}'.format(idx, err_cnt)
    # print(msg)
    logging.info(msg)

err_stats = pd.Series(err_stats)
err_stats = err_stats[err_stats > 0]
accuracy = 1 - sum(err_stats.index * err_stats) / len(data) / conway.size
logging.critical('Error Counts:\n{}'.format(err_stats))
logging.critical('Accuracy = {}'.format(accuracy))
print('All are done.')
