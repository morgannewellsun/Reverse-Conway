import tensorflow as tf
import numpy as np
import pandas as pd
import time
import logging
import pathlib
from datetime import datetime
from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn
from components.reversega import ReverseGa


max_csv_rows = 100           # maximum number of rows loaded
delta_1_only = True        # Load only the model for delta = 1
cnn_path = '../../Reverse-Conway/pretrained_models/initial_baseline_delta_'
# cnn_path = '../../Reverse-Conway/pretrained_models/supervised_baseline_delta_'
rand_seed = 0             # Used in genetic algorithm ReverseGa
ga_pop_size = 20
ga_max_iters = 20
ga_cross = 1
ga_mutate = 1
ga_mut_div = 100
status_freq = 100          # Report frequency in terms of number of games
track_details = True
kaggle_test_file = '../../gamelife_data/kaggle/test.csv'
output_dir = '../../gamelife_data/output/'
# If False, bypass CNN results to save load time. Use raondom initial states.
use_cnn = True

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


def save_results(all_results):
    if len(all_results[0]) > 2:
        data = pd.DataFrame(all_results, columns = [
            'Game Index', 'Delta', 'Target Lives', 'CNN Lives', 'CNN Errors',
            'GA Lives', 'GA Errors', 'Target State', 'CNN Start', 'GA Start'])
    else:
        data = pd.DataFrame(all_results, coloums = ['Game Index', 'Start'])
    data.to_csv(output_dir + 'results.csv')
    pd.DataFrame([
        ['cnn_path', cnn_path],
        ['max_csv_rows', max_csv_rows],
        ['rand_seed', rand_seed],
        ['ga_pop_size', ga_pop_size],
        ['ga_max_iters', ga_max_iters],
        ['ga_cross', ga_cross],
        ['ga_mutate', ga_mutate],
        ['ga_mut_div', ga_mut_div],
        ['delta_1_only', delta_1_only],
        ['start_time', start_time],
        ['end_time', end_time]
        ], columns=('key', 'value')
        ).to_csv(output_dir + 'config.csv')
    


mylog('Reverse Conway started.')
prev_t = time.time()
start_time = datetime.now().isoformat(' ')

#### Load CNN solvers from files.
cnn_solver = list()
if use_cnn:
    for j in range(1, 6):
        path_to_saved_model = cnn_path + str(j)
        cnn = tf.keras.models.load_model(path_to_saved_model, compile=False)  # compile=True will fail!
        cnn_solver.append(cnn)
        if delta_1_only:
            break
    mylog('CNN models loaded after {}'.format(timing()))


#### Load Kaggle test files
data = pd.read_csv(kaggle_test_file, index_col=0, dtype='int', nrows=max_csv_rows)
mylog('Kaggle file loaded after {}'.format(timing()))


#### Apply GA to improve.
np.random.seed(rand_seed) 
conway = BinaryConwayForwardPropFn(numpy_mode=True, nrows=25, ncols=25)
ga = ReverseGa(conway, pop_size=ga_pop_size, max_iters=ga_max_iters,
               crossover_rate=ga_cross, mutation_rate=ga_mutate,
               mut_div = ga_mut_div, tracking=track_details)

all_results = []
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
with open(output_dir + 'details.txt', 'w') as detail_file:
    for idx, row in data.iterrows():
        delta = row[0]
        if delta_1_only and (not delta == 1):
            continue
        tf_arr = np.array(row[1:]).astype(np.float32).reshape((1, 25, 25, 1))
        if use_cnn:
            solv_1 = cnn_solver[delta-1](tf_arr).numpy()
        else:
            solv_1 = None
        res = ga.refine_cnn(idx, delta, np.array(row[1:].to_list()), solv_1)
        all_results.append(res)
        if track_details:
            detail_file.write('Game {} with delta {}:\n{}\n\n'.format(
                idx, delta, ga.summary()))
        if idx % status_freq == 0:
            mylog('Completed game {} after {}.'.format(idx, timing()))

end_time = datetime.now().isoformat(' ')
save_results(all_results)
mylog('Solver completed.')

