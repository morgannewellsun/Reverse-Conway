# Conway inverse solution using single processor.

# BEGIN USER SETTINGS

cnn_paths = (
    '../../Reverse-Conway/pretrained_models/crossfade_baseline_delta_1',
    # '../../Reverse-Conway/pretrained_models/initial_baseline_delta_1',
    '../../Reverse-Conway/pretrained_models/initial_baseline_delta_2',
    '../../Reverse-Conway/pretrained_models/initial_baseline_delta_3',
    '../../Reverse-Conway/pretrained_models/initial_baseline_delta_4',
    '../../Reverse-Conway/pretrained_models/initial_baseline_delta_5' )
# cnn_path = '../../Reverse-Conway/pretrained_models/supervised_baseline_delta_'
rand_seed = 0             # Used in genetic algorithm ReverseGa
ga_pop_size = 100
ga_static_size = 30       # GA initial population from the static prob
ga_max_iters = 100
ga_cross = 0.7              # GA cross ratio
ga_mutate = 0.7             # GA mutation population ratio
ga_mut_div = 100          # GA cell mutation probability is 1/ga_mut_div
ga_max_stales = 2          # GA maximum iterations without improvements
ga_save_states = False        # Should we save CNN state, GA state, and end state?
status_freq = 100          # Report frequency in terms of number of games
track_details = False
kaggle_test_file = '../../gamelife_data/kaggle/test.csv'
output_dir = '../../gamelife_data/output/'
# If False, bypass CNN results to save load time. Use raondom initial states.
use_cnn = True
# The following settings restricts to only a selected subset of data to test.
deltaset = {1}        # Load only the model for specified deltas.
game_idx_min = 0         # Kaggle test game indices from 50000 to 99999.
game_idx_max = 50100      # To test for 1000 rows, use 51000
stepwise = False          # If true, also run iteratively of 1-delta CNN.

# END USER SETTINGS


import tensorflow as tf
import numpy as np
import pandas as pd
import time
import logging
import pathlib
from datetime import datetime
from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn
from old.reversega import ReverseGa
from old.cnnman import CnnMan
from old.revconwayreport import post_run_report


def mylog(msg):
    # tensorflow customized the logggin. Don't want to fight it.
    logging.critical(datetime.now().isoformat(' ') + ' ' + msg)


def timing():
    global prev_t
    t = time.time()
    t_sec = round(t - prev_t)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    prev_t = t
    return '{}:{}:{}'.format(t_hour, t_min, t_sec)


result_header = [
    'Game Index', 'Delta', 'Target Lives', 'CNN Lives', 'CNN Errors',
    'GA Lives', 'GA Errors']
if ga_save_states:
    result_header.extend(['Target State', 'CNN Start', 'GA Start'])


def save_results(all_submissions, all_results):
    cols = ['id']
    cols.extend(['start_' + str(j) for j in range(625)])
    pd.DataFrame(all_submissions, columns=cols).to_csv(output_dir + 'submission.csv', index=False)

    # Record basic settings for later review with results.
    pd.DataFrame([
        ['cnn_paths', cnn_paths],
        ['deltaset', deltaset],
        ['ga_cross', ga_cross],
        ['ga_mut_div', ga_mut_div],
        ['ga_max_iters', ga_max_iters],
        ['ga_max_stales', ga_max_stales],
        ['ga_mutate', ga_mutate],
        ['ga_pop_size', ga_pop_size],
        ['ga_save_states', ga_save_states],
        ['ga_static_size', ga_static_size],
        ['game_idx_min', game_idx_min],
        ['game_idx_max', game_idx_max],
        ['rand_seed', rand_seed],
        ['stepwise', stepwise],
        ['track_details', track_details],
        ['use_cnn', use_cnn],
        ['start_time', start_time],
        ['end_time', end_time]
        ], columns=('key', 'value')
        ).to_csv(output_dir + 'config.csv')
    if len(all_results) == 0:
        print('No results were generated.')
        return
    data = pd.DataFrame(all_results, columns = result_header)
    data.to_csv(output_dir + 'results.csv')
    # Generate more statistical reports based on the above file.
    post_run_report(output_dir)



mylog('Reverse Conway started.')
prev_t = time.time()
start_time = datetime.now().isoformat(' ')

#### Load CNN solvers from files.
cnn_solver = dict()
if use_cnn:
    for j in deltaset:
        # cnn = tf.keras.models.load_model(cnn_paths[j-1], compile=False)  # compile=True will fail!
        cnn = tf.saved_model.load(cnn_paths[j-1])
        cnn_solver[j] = cnn
    mylog('CNN models loaded after {}'.format(timing()))
    cnn_manager = CnnMan(cnn_solver, stepwise)


#### Load Kaggle test files
data = pd.read_csv(kaggle_test_file, index_col=0, dtype='int')
mylog('Kaggle file loaded after {}'.format(timing()))


#### Apply GA to improve.
np.random.seed(rand_seed) 
conway = BinaryConwayForwardPropFn(numpy_mode=True, nrows=25, ncols=25)
ga = ReverseGa(conway, pop_size=ga_pop_size, max_iters=ga_max_iters,
               crossover_rate=ga_cross, mutation_rate=ga_mutate,
               mut_div = ga_mut_div, max_stales=ga_max_stales,
               tracking=track_details, save_states=ga_save_states)

all_results = []
all_submissions = []
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
with open(output_dir + 'details.txt', 'w') as detail_file:
    for idx, row in data.iterrows():
        if idx < game_idx_min:
            continue
        if idx > game_idx_max:
            break
        delta = row[0]
        if not delta in deltaset:
            continue
        
        tf_arr = np.array(row[1:]).astype(np.float32).reshape((1, 25, 25, 1))
        if use_cnn:
            solv_1 = cnn_manager.revert(tf_arr, delta, ga_pop_size, ga_static_size)
        else:
            solv_1 = None
        submission, res = ga.revert(idx, delta, tf_arr.astype(bool), solv_1)
        all_submissions.append(submission)
        all_results.append(res)
        if track_details:
            res_dict = dict(zip(result_header[:7], res[:7]))
            detail_file.write('Details for {}:\n{}\n\n'.format(
                res_dict, ga.summary()))
        if idx % status_freq == 0:
            mylog('Completed game {} after {}.'.format(idx, timing()))

end_time = datetime.now().isoformat(' ')
save_results(all_submissions, all_results)
mylog('Solver completed.')

