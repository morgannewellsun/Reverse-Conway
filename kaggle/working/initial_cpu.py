# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import tensorflow as tf
import time
import pathlib
from datetime import datetime

# %% [code]
# Conway's game logic

class BinaryConwayForwardPropFn:

    def __init__(self, numpy_mode=False, nrows=25, ncols=25):
        self._numpy_mode = numpy_mode
        self.nrows = nrows
        self.ncols = ncols
        self._moore_offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i != 0 or j != 0)]

    def __call__(self, inputs, delta = 1):
        # inputs is a np array of at least 3D, usually 4D, of shape:
        # (batch_size, game board width, game board height, 1).
        # outputs will be of the same shape as inputs.
        # For an example of use, see
        # Reverse-Conway/src/data/tests/verify_kaggle_training.py
        outputs = inputs
        for _ in range(delta):
            outputs = self._one_delta(outputs)
        return outputs


    def _one_delta(self, inputs):
        if self._numpy_mode:
            neighbors = [np.roll(inputs, shift, (-3, -2)) for shift in self._moore_offsets]
            live_neighbor_counts = np.count_nonzero(neighbors, axis=0)
            two_live_neighbors = np.equal(live_neighbor_counts, 2)
            three_live_neighbors = np.equal(live_neighbor_counts, 3)
            outputs = np.logical_or(three_live_neighbors, np.logical_and(two_live_neighbors, inputs))
        else:
            neighbors = [tf.roll(inputs, shift, (-3, -2)) for shift in self._moore_offsets]
            live_neighbor_counts = tf.math.count_nonzero(neighbors, axis=0)
            two_live_neighbors = tf.math.equal(live_neighbor_counts, 2)
            three_live_neighbors = tf.math.equal(live_neighbor_counts, 3)
            outputs = tf.math.logical_or(three_live_neighbors, tf.math.logical_and(two_live_neighbors, inputs))
        return outputs

# %% [code]
# Generate initial reverse same guesses using CNN.

class CnnMan:
    # A CNN solver manager. It produces solutions to Conway games using CNN.
    
    def __init__(self, cnn_reverters, stepwise):
        # Arg cnn_reverters is a dictionary from int (delta)
        # to a CNN solver. A solver accepts array of np.float32.
        self._cnn_rerverters = cnn_reverters
        self._stepwise = stepwise


    def _revert_many(self, model, stop_states):
        # Use CNN to revert many boards by 50% threshold.
        # Arg stop_states is an array of size
        # (number of boards, nrows, ncols, 1).
        cnn_result = model(stop_states.astype(np.float32)).numpy()
        return cnn_result >= 0.5
    
    
    def _revert_static(self, model, stop_state, popsize):
        # Use CNN to revert a single board.
        # Arg popsize is the number of output game boards.
        # Arg stop_state is array of size (1, nrows, ncols, 1).
        cnn_result = model(stop_state).numpy()
        sorted_probs = sorted(cnn_result.flatten())
        life50 = (cnn_result < 0.5).sum()
        half_pop = int(popsize / 2)
        if life50 - half_pop < 0:
            selected = range(popsize)
        elif life50 - half_pop + popsize < np.prod(stop_state.shape):
            selected = range(life50 - half_pop, life50 - half_pop + popsize)
        else:
            selected = range(-popsize, 0)
        # This is a list of 1D 0/1 arrays representing the boards from CNN.
        return np.array([(cnn_result[0] > sorted_probs[j]) for j in selected])


    def _revert_dynamic(self, model, stop_state, popsize):
        # Use CNN to revert a single board by detal=1.
        # Arg popsize is the number of output game boards.
        # Arg stop_state is array of size (1, nrows, ncols, 1).
        cnn_result = model(stop_state).numpy()
        s = stop_state.shape
        return np.random.binomial(1, cnn_result, (popsize, s[1], s[2], 1))

    
    def revert(self, stop_state, delta, popsize, static_size):
        # Return initial game boards as an array of bool with size
        # (count, width, height, 1), with count at least popsize.
        
        # This is CNN model to revert in one shot.
        model_d = self._cnn_rerverters[delta]
        unit_size = popsize - static_size
        cnn_results = []
        cnn_results.extend(self._revert_static(model_d, stop_state, static_size))
        if unit_size > 0:
            model_1 = None
            if self._stepwise and delta > 1:
                # This uses CNN 1-step revert model iteratively.
                model_1 = self._cnn_rerverters[1]
                unit_size = math.ceil(unit_size / 2)
            # Revert delta steps back in one shot.
            model_res = self._revert_dynamic(model_d, stop_state, unit_size)
            cnn_results.extend(model_res)
            if model_1:
                # Revert 1 step in each iteration.
                res = self._revert_dynamic(model_1, stop_state, unit_size)
                for _ in range(delta - 1):
                    res = self._revert_many(model_1, res)
                cnn_results.extend(res)
        return np.array(cnn_results)


# %% [code]
# Enhance revert game results using genetic algorithm.

class ReverseGa:
    
    def __init__(self, conway:BinaryConwayForwardPropFn,
                 pop_size = 10, max_iters = 10,
                 mut_div = 10, max_stales = 2, tracking = True,
                 save_states = True):
        # Arg mut_div: probability of mutation is 1/mut_div
        self.conway = conway
        self._chromo_len = conway.nrows * conway.ncols
        self.pop_size = pop_size
        self._max_iters = max_iters
        self._mutation_div = mut_div
        self._max_stales = max_stales     # max iterations without improvements.
        self._tracking = tracking
        self._save_states = save_states
        self._offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]


    def revert(self, game_idx, delta, stop_state, guess = None):
        """ Arguments:
            stop_state is the 4D array (bool) representation of the stop state,
            of shape (1, game_board_width, game_board_height, 1)
            guess is either None or a 4D array of shape
            (batch_size, game_board_width, game_board_height, 1).
            Return:
            A tuple of two:
            initial state represented by a 4D array of bool
            and the number of errors, which is the count of
            number of end cells from the initial state different from stop_state.
        """
        self._delta = delta
        self._target = stop_state
        self._reset(guess)
        self._select()
        self._track()      # Generation 0 is done.
        cnn_guess = self._curr_pop[0]
        cnn_lives = cnn_guess.sum()
        cnn_errors = self._best_error
        
        while self._gen_idx < self._max_iters:
            self._gen_idx += 1
            if self._best_error == 0:
                break
            self._mutate()
            self._crossover()
            self._select()
            self._track()
            if (self._gen_idx - self._best_gen > self._max_stales
                and self._gen_idx - self._worst_gen > self._max_stales):
                break

        ga_result = self._curr_pop[0]
        target_lives = stop_state.sum()
        ga_lives = ga_result.sum()
        submission = [game_idx]
        submission.extend(ga_result.flatten().astype(int).tolist())
        if self._save_states:
            more = [game_idx, delta, target_lives, cnn_lives, cnn_errors,
                    ga_lives, self._best_error, 
                    ''.join(map(str, stop_state.flatten().astype(int).tolist())),
                    ''.join(map(str, cnn_guess.flatten().astype(int).tolist())),
                    ''.join(map(str, ga_result.flatten().astype(int).tolist())) ]
        else:
            more = [game_idx, delta, target_lives, cnn_lives, cnn_errors,
                    ga_lives, self._best_error]
        return submission, more
            

    def _reset(self, guess):
        # set up generation 0.
        self._curr_pop = None      # 4D np.array. _curr_pop[0] is the best, _curr_pop[-1] is the worst.
        self._gen_idx = 0                # Current generation index.
        self._best_gen = 0               # The generate giving the best chromo
        self._worst_gen = 0            # The generation whose worst is the best.
        self._best_error = self._chromo_len     # The smallest error so far
        self._worst_error = self._chromo_len
        if self._tracking:
            self._report = list()

        # Generation 0 start from building self._mutants + _babies.
        empty_state = np.array([False] * self._chromo_len).reshape(self._target.shape)
        self._mutants = np.concatenate((empty_state, self._target))
        self._target_diff = self.conway(self._target, self._delta) ^ self._target
        self._target_err = np.array(self._target_diff.sum(axis=(1,2,3)))
        if guess is None:
            # Not enough intial guesses are supplied. Use random states.
            sz = (self.pop_size, self.conway.nrows, self.conway.ncols, 1)
            self._babies = np.random.randint(2, size=sz).astype(bool)
        else:
            self._babies = guess


    def _mutate(self):
        # The resulting board has 1 / self._mutation_div fraction being live cells.
        muter = (np.random.randint(self._mutation_div, size=self._curr_pop.shape)
                 / (self._mutation_div - 1)).astype(int).astype(bool)
        # Outside this area, we don't mutate any cells.
        target_area = np.any([np.roll(self._diffs, shift, (-3, -2)) for shift in self._offsets], axis=0)
        muter &= target_area
        self._mutants = self._curr_pop ^ muter
        # In case muter[j] are all zeros, no mutation happens. Remove it.
        self._mutants = self._mutants[self._mutants.sum(axis=(1,2,3))>0]


    def _crossover(self):
        idx = np.random.permutation(len(self._curr_pop))
        nco = int(len(self._curr_pop)/2)
        dads = self._curr_pop[idx[:nco]]
        moms = self._curr_pop[idx[nco:(2*nco)]]
        swapper = np.random.randint(low=0, high=2, size=(
            nco, self.conway.nrows, self.conway.ncols, 1)).astype(bool)
        complim = ~swapper
        self._babies = np.concatenate((
            (dads & swapper) | (moms & complim),
            (dads & complim) | (moms & swapper) ))
        self._babies = self._babies[self._babies.sum(axis=(1,2,3))>0]


    def _add_newpop(self, newpop):
        # Add newpop into the current population.
        # Arg newpop must be 4D np.array.
        # The new population after Conway game forward is compared to the target.
        # The differences and the number of errors are recorded.

        # self._diffs is 4D np.array of differences from target
        # self._errors is 1D array of error count.

        newdiff = self.conway(newpop, self._delta) ^ np.repeat(self._target, len(newpop), axis=0)
        newerr = np.array(newdiff.sum(axis=(1,2,3)))
        if self._curr_pop is None:
            self._curr_pop = newpop
            self._diffs = newdiff
            self._errors = newerr
        else:
            self._curr_pop = np.concatenate((self._curr_pop, newpop))
            self._diffs = np.concatenate((self._diffs, newdiff))
            self._errors = np.concatenate((self._errors, newerr))
        pre_cnt = len(self._curr_pop)
        if self._gen_idx % 10 == 9:
            # Every 10 rounds, we purge those duplicated states. This is expensive.
            self._curr_pop, idx = np.unique(self._curr_pop, axis=0, return_index=True)
            self._diffs = self._diffs[idx]
            self._errors = self._errors[idx]
        self._trims = pre_cnt - len(self._curr_pop)


    def _select(self):
        """ Trim down current population to pop_size
        """
        # The order of adding mutants and babies is important.
        # Later the statistics are computed assuming this addition order.
        self._add_newpop(np.concatenate((self._mutants, self._babies)))
        # Make sure the best and the worst chromos
        # after the selection are correctly placed.
        last_idx = min(self.pop_size, len(self._curr_pop))
        self._selects = np.argpartition(self._errors, (0, last_idx-1))
        self._selects = self._selects[0:last_idx]
        self._curr_pop = self._curr_pop[self._selects]
        self._diffs = self._diffs[self._selects]
        self._errors = self._errors[self._selects]
        best = self._errors[0]
        if best < self._best_error:
            self._best_error = best
            self._best_gen = self._gen_idx
        worst = self._errors[-1]
        if worst < self._worst_error:
            self._worst_error = worst
            self._worst_gen = self._gen_idx
        self._curr_pop = np.concatenate((self._curr_pop, self._target))
        self._diffs = np.concatenate((self._diffs, self._target_diff))
        self._errors = np.concatenate((self._errors, self._target_err))
        

    def _track(self):
        # Track statistics in this generation for review.
        if not self._tracking:
            return
        n0 = self.pop_size
        n1 = n0 + len(self._mutants)
        if self._selects[0] < n0:
            src = 'P'
        elif self._selects[0] < n1:
            src = 'M'
        else:
            src = 'B'
        self._report.append([
            len(self._mutants), len(self._babies),
            self._trims,    # number of duplicates that are trimmed
            sum(self._selects < n0),  # survivals
            sum((self._selects >= n0) & (self._selects < n1)),  # selected mutants
            sum(self._selects >= n1),   # selected babies (from crossovers)
            self._best_error, self._worst_error,
            self._best_gen, self._worst_gen, src
            ])


    def summary(self):
        return pd.DataFrame(
            self._report, columns=(
                'mut', 'bab', 'dup', 'p_in', 'm_in', 'b_in',
                'min_e', 'max_e', 'best', 'worst', 'src'))


# %% [code]
# Utility functions

def mylog(title):
    global prev_t
    t = time.time()
    t_sec = round(t - prev_t)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    prev_t = t
    print('{} {} after {}:{}:{}'.format(datetime.now().isoformat(' '), title, t_hour, t_min, t_sec))


    
def save_results(all_submissions, all_results, start_time, end_time):
    if len(all_results) == 0:
        print('No results were generated.')
        return

    # Submission file following Kaggle's format
    cols = ['id']
    cols.extend(['start_' + str(j) for j in range(625)])
    pd.DataFrame(all_submissions, columns=cols).to_csv(output_dir + 'submission.csv', index=False)

    with pd.ExcelWriter(output_dir + 'run_stats.xlsx', engine='xlsxwriter') as writer:
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
            ).to_excel(writer, sheet_name='config')
        data = pd.DataFrame(all_results, columns = result_header)
        data.to_excel(writer, sheet_name='result')

        # Generate more statistical reports based on the above data.
        game_size = 25 * 25
        # statistics by errors
        err_col = ['delta ' + str(j) for j in range(6)]
        err_stats = pd.DataFrame([[0]*6]*game_size, columns=err_col)
        # statistics by number of lives at the end state
        liv_stats = pd.DataFrame([[0]*3]*game_size, columns=(
            'count', 'cnn_fails', 'ga_fails'))
        del_stats = pd.DataFrame([[0]*5]*6, columns=(
            'count', 'cnn_hits', 'ga_hits', 'cnn_fails', 'ga_fails'))

        for j, row in data.iterrows():
            (game_index, delta, target_lives, cnn_lives, cnn_errors,
             ga_lives, ga_errors) = map(int, row[:7])

            err_stats.iloc[ga_errors, delta] += 1
            liv_stats.iloc[target_lives, 0] += 1
            liv_stats.iloc[target_lives, 1] += cnn_errors
            liv_stats.iloc[target_lives, 2] += ga_errors
            del_stats.iloc[delta, 0] += 1
            del_stats.iloc[delta, 3] += cnn_errors
            del_stats.iloc[delta, 4] += ga_errors
            if cnn_errors == 0:
                del_stats.iloc[delta, 1] += 1
            if ga_errors == 0:
                del_stats.iloc[delta, 2] += 1

        err_stats['total'] = err_stats.sum(axis=1)
        err_stats = err_stats.loc[err_stats['total']>0, :]
        err_stats.to_excel(writer, sheet_name='errors')

        liv_stats = liv_stats.loc[liv_stats['count']>0, :]
        liv_stats['cnn_accuracy'] = 1 - liv_stats['cnn_fails'] / liv_stats['count'] / game_size
        liv_stats['ga_accuracy'] = 1 - liv_stats['ga_fails'] / liv_stats['count'] / game_size
        liv_stats.to_excel(writer, sheet_name='lives')

        del_stats = del_stats[del_stats.index > 0]
        del_stats['cnn_accuracy'] = 1 - del_stats['cnn_fails'] / del_stats['count'] / game_size
        del_stats['ga_accuracy'] = 1 - del_stats['ga_fails'] / del_stats['count'] / game_size
        del_stats.to_excel(writer, sheet_name='deltas')


# %% [code]
# USER SETTINGS# USER SETTINGS

model_root_dir = '../input/conway/cnn_models/'
cnn_paths = (
    model_root_dir + 'delta_1',
    model_root_dir + 'delta_2',
    model_root_dir + 'delta_3',
    model_root_dir + 'delta_4',
    model_root_dir + 'delta_5' )
kaggle_test_file = '../input/conways-reverse-game-of-life-2020/test.csv'
output_dir = './'

rand_seed = 0             # Used in genetic algorithm ReverseGa
ga_pop_size = 120
ga_static_size = 20       # GA initial population from the static prob
ga_max_iters = 100
ga_cross = 1              # GA cross ratio.
ga_mutate = 1             # GA mutation population ratio. Fixed.
ga_mut_div = 100          # GA cell mutation probability is 1/ga_mut_div
ga_max_stales = 2          # GA maximum iterations without improvements
ga_save_states = False        # Should we save CNN state, GA state, and end state?
status_freq = 200          # Report frequency in terms of number of games
track_details = False
# If False, bypass CNN results to save load time. Use raondom initial states.
use_cnn = True
# The following settings restricts to only a selected subset of data to test.
deltaset = {1,2,3,4,5}        # Load only the model for specified deltas. To load all, use {1,2,3,4,5}
game_idx_min = 0         # Kaggle test game indices from 50000 to 99999.
game_idx_max = 51000      # To test for 1000 rows, use 51000. Use math.inf for all
stepwise = False          # If true, also run iteratively of 1-delta CNN. This is time consuming.


# %% [code]
# Load CNN models, test file, and set up GA.# Load CNN models, test file, and set up GA.

result_header = [
    'Game Index', 'Delta', 'Target Lives', 'CNN Lives', 'CNN Errors',
    'GA Lives', 'GA Errors']
if ga_save_states:
    result_header.extend(['Target State', 'CNN Start', 'GA Start'])

prev_t = time.time()
mylog('Reverse Conway started.')
start_time = datetime.now().isoformat(' ')

#### Load CNN solvers from files.
cnn_solver = dict()
if use_cnn:
    for j in deltaset:
        # cnn = tf.keras.models.load_model(cnn_paths[j-1], compile=False)  # compile=True will fail!
        cnn = tf.saved_model.load(cnn_paths[j-1])
        cnn_solver[j] = cnn
    mylog('CNN models loaded')
    cnn_manager = CnnMan(cnn_solver, stepwise)


#### Load Kaggle test files
data = pd.read_csv(kaggle_test_file, index_col=0, dtype='int')
mylog('Kaggle file loaded')


#### Apply GA to improve.
np.random.seed(rand_seed) 
conway = BinaryConwayForwardPropFn(numpy_mode=True, nrows=25, ncols=25)
ga = ReverseGa(conway, pop_size=ga_pop_size, max_iters=ga_max_iters,
               mut_div = ga_mut_div, max_stales=ga_max_stales,
               tracking=track_details, save_states=ga_save_states)


# %% [code]
# Actual run

mylog('GA run started')
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
            mylog('Completed game {}.'.format(idx))

save_results(all_submissions, all_results, start_time, datetime.now().isoformat(' '))
mylog('Conway solver completed.')
