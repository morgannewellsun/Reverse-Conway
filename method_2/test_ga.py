
# Attempt to solve the reverse game by genetic algorithm.

import pandas as pd
import logging
import random
from gamelib.reversega import ReverseGa
from gamelib.logutil import init_game_log, kaggle_root, timing
from gamelib.conwaymap import ConwayMap

init_game_log('Reverse solver by genetic algorithm', lvl = logging.ERROR)

# Max number of rows from the train/test.csv files.
# Kaggle supplied training file has 50,001 lines
# Use small numbers to test first.
max_csv_rows = 50
report_freq = 1000

random.seed(0)

conway = ConwayMap(nrows=25, ncols=25)
ga = ReverseGa(conway, pop_size=5, max_iters=5,
               crossover_rate=1, mutation_rate=0.5, tracking=False)
err_stats = [0] * conway.size
count_stats = [0] * conway.size
fail_stats = [0] * conway.size
delta_rows= [0] * 6
delta_errors = [0] * 6

data = pd.read_csv(kaggle_root + 'test.csv', index_col=0, dtype='str', nrows=max_csv_rows)
for idx, row in data.iterrows():
    delta = int(row[0])
    lives = row[1:].to_list().count('1')
    state = conway.str_to_bin(''.join(row[1:]))
    err_cnt = ga.revert(delta, state)
    err_stats[err_cnt] += 1
    count_stats[lives] += 1
    fail_stats[lives] += err_cnt
    delta_rows[delta] += 1
    delta_errors[delta] += err_cnt
    msg = 'Run {} has {} lives with {} errors'.format(idx, lives, err_cnt)
    # print(msg)
    logging.info(msg)
    if idx % report_freq == 0:
        t = timing()
        print('Processed row {} after {} since last reported.'.format(idx, t))

err_stats = pd.Series(err_stats)
err_stats = err_stats[err_stats > 0]
accuracy = 1 - sum(err_stats.index * err_stats) / len(data) / conway.size
logging.critical('Error Counts:\n{}'.format(err_stats))
logging.critical('Accuracy = {}'.format(accuracy))
den_df = pd.DataFrame({'count': count_stats, 'fails': fail_stats})
den_df['ratio'] = 1 - den_df['fails'] / den_df['count'] / conway.size
den_df = den_df.loc[den_df['count'] > 0, :]
den_df.index.rename('lives')
logging.critical('Stats based on lives\n{}'.format(den_df))
delta_df = pd.DataFrame({'count': delta_rows, 'fails': delta_errors})
delta_df['ratio'] = 1 - delta_df['fails'] / delta_df['count'] / conway.size
delta_df = delta_df.loc[delta_df['count'] > 0, :]
logging.critical('Stats by delta\n{}'.format(delta_df))
print('All are done.')
