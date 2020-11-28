# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
        # (popsize, batch_size, game board width, game board height, 1).
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
    
    def __init__(self, cnn_reverters, stepwise, prob_span=0.2):
        # Arg cnn_reverters is a dictionary from int (delta)
        # to a CNN solver. A solver accepts array of np.float32.
        self._cnn_reverters = cnn_reverters
        self._stepwise = stepwise
        self._prob_span = prob_span


    def _revert_many(self, model, stop_states):
        # Use CNN to revert many boards by 50% threshold.
        # Arg stop_states is an array of size
        # (number of boards, nrows, ncols, number of guesses).
        cnn_result = model(stop_states.astype(np.float32)).numpy()
        return cnn_result >= 0.5
    
    
    def _revert_static(self, model, stop_state, popsize):
        # Use CNN to revert a single board.
        # Arg popsize is the number of output game boards.
        # Arg stop_state is array of size (1, nrows, ncols, batches).
        # Return np array of (batches, 25, 25, popsize)
        if popsize == 0:
            return []
        cnn_result = model(stop_state).numpy()        # shape = (1, 25, 25, batches)
        bars = 0.5-self._prob_span/2 + np.array(range(popsize)) * self._prob_span/(popsize-1)
        x = np.array([(cnn_result[0] > p) for p in bars])     # shape = (popsize, 25, 25, batches)
        return x


    def _revert_dynamic(self, model, stop_state, popsize):
        # Use CNN to revert a single board by delta=1.
        # Arg popsize is the number of output game boards.
        # Arg stop_state is array of size (1, nrows, ncols, batches).
        # Return np array of (batches, 25, 25, popsize)
        if popsize == 0:
            return []
        cnn_result = model(stop_state).numpy()  # shape = (1, 25, 25, batches)
        cnn_result = cnn_result[0]
        s = cnn_result.shape
        x = np.random.binomial(1, cnn_result, (popsize, *s))
        return x    # np.moveaxis(x, (0,1,2,3), (3,1,2,0))

    
    def revert(self, stop_state, delta, static_1, static_n, dynamic_1, dynamic_n):
        # Return initial game boards as an array of bool with size
        # (count, width, height, batches), with count at least popsize.
        
        # This is CNN model to revert in one shot.
        model_d = self._cnn_reverters[delta]
        cnn_results = []
        if delta == 1:
            cnn_results.extend(self._revert_static(model_d, stop_state, static_1 + static_n))
            cnn_results.extend(self._revert_dynamic(model_d, stop_state, dynamic_1 + dynamic_n))
        else:
            cnn_results.extend(self._revert_static(model_d, stop_state, static_n))
            cnn_results.extend(self._revert_dynamic(model_d, stop_state, dynamic_n))
            cnn_1 = []
            model_1 = self._cnn_reverters[1]
            cnn_1.extend(self._revert_static(model_1, stop_state, static_1))
            cnn_1.extend(self._revert_dynamic(model_1, stop_state, dynamic_1))
            for _ in range(delta - 1):
                cnn_1 = self._revert_many(model_1, cnn_1)
            cnn_results.extend(cnn_1)
        return np.array(cnn_results)


# %% [code]
# Enhance revert game results using genetic algorithm.

class ReverseGa:
    
    def __init__(self, conway:BinaryConwayForwardPropFn,
                 pop_size = 10, max_iters = 10,
                 crossover_rate = 1, mutation_rate = 0.5,
                 mut_div = 10, tracking = True,
                 save_states = True):
        # Arg mut_div: probability of mutation is 1/mut_div
        self.conway = conway
        self.pop_size = pop_size
        self._max_iters = max_iters
        self._nmutations = int(pop_size * mutation_rate)
        self._mutation_div = mut_div
        self._ncrossover = int(pop_size * crossover_rate / 2)
        self._tracking = tracking
        self._save_states = save_states
        self._offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]


    def revert(self, game_idx, delta, stop_state, guess):
        """ Arguments:
            stop_state is the 4D array (bool) representation of the stop state,
            of shape (1, game_board_width, game_board_height, batches)
            guess is a 4D array of shape
            (popsize, game_board_width, game_board_height, batch_size).
            Return:
            A tuple of two:
            array of size (1, 25, 25, batches)
            and a list of statistical information.
        """
        self._chromo_len = np.product(stop_state.shape)
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

        # Generation 0 starts from building self._mutants + _babies.
        empty_state = np.array([False] * self._chromo_len).reshape(self._target.shape)
        self._mutants = np.concatenate((empty_state, self._target))
        if guess is None:
            # Not enough intial guesses are supplied. Use random states.
            sz = (self.pop_size, self.conway.nrows, self.conway.ncols, 1)
            self._babies = np.random.randint(2, size=sz).astype(bool)
        else:
            self._babies = guess


    def _mutate(self):
        c = np.random.choice(len(self._curr_pop), replace=False,
                             size=min(self._nmutations, len(self._curr_pop)))
        chromos = self._curr_pop[c]   # shape = (selected, 25, 25, batches)
        # The resulting board has 1 / self._mutation_div fraction being live cells.
        muter = (np.random.randint(self._mutation_div, size=chromos.shape)
                 / (self._mutation_div - 1)).astype(int).astype(bool)
        # Outside this area, we don't mutate any cells.
        target_area = np.any([np.roll(self._diffs[c], shift, (-3, -2)) for shift in self._offsets], axis=0)
        muter &= target_area
        self._mutants = chromos ^ muter


    def _crossover(self):
        idx = np.random.choice(len(self._curr_pop), size=2*self._ncrossover, replace=True)
        dads = self._curr_pop[idx[:self._ncrossover]]
        moms = self._curr_pop[idx[self._ncrossover:]]
        swapper = np.random.randint(low=0, high=2, size=dads.shape).astype(bool)
        complim = ~swapper
        self._babies = np.concatenate((
            (dads & swapper) | (moms & complim),
            (dads & complim) | (moms & swapper) ))


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
            ['batch_size', batch_size],
            ['cnn_paths', cnn_paths],
            ['deltaset', deltaset],
            ['ga_cross', ga_cross],
            ['ga_mut_div', ga_mut_div],
            ['ga_max_iters', ga_max_iters],
            ['ga_mutate', ga_mutate],
            ['ga_dynamic_1', ga_dynamic_n],
            ['ga_dynamic_n', ga_dynamic_n],
            ['ga_static_1', ga_static_1],
            ['ga_static_n', ga_static_n],
            ['ga_pop_size', ga_pop_size],
            ['ga_save_states', ga_save_states],
            ['ga_static_size', ga_static_size],
            ['game_idx_min', game_idx_min],
            ['game_idx_max', game_idx_max],
            ['rand_seed', rand_seed],
            ['stepwise', stepwise],
            ['track_details', track_details],
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
        
        pd.DataFrame(mytime, index=['vallue']).T.to_excel(writer, sheet_name='timing')



def tic(key):
    myprev[key] = time.time()



def toc(key):
    mytime[key] += time.time() - myprev[key]



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

batch_size = 10           # Delta group size
rand_seed = 0             # Used in genetic algorithm ReverseGa
ga_static_1 = 10       # GA initial population from the static prob, step-wise
ga_static_n = 10       # GA initial population from the static prob, direct solver
ga_dynamic_1 = 10       # GA initial population from the dynamic prob, step-wise
ga_dynamic_n = 10       # GA initial population from the dynamic prob, direct solver
ga_pop_size = ga_static_1 + ga_static_n + ga_dynamic_1 + ga_dynamic_n
ga_max_iters = 100
ga_cross = 0.7              # GA cross ratio
ga_mutate = 0.7             # GA mutation population ratio
ga_mut_div = 100          # GA cell mutation probability is 1/ga_mut_div
ga_save_states = False        # Should we save CNN state, GA state, and end state?
status_freq = 200          # Report frequency in terms of number of games
track_details = False
# The following settings restricts to only a selected subset of data to test.
deltaset = {1}        # Load only the model for specified deltas. To load all, use {1,2,3,4,5}
game_idx_min = 50003         # Kaggle test game indices from 50000 to 99999.
game_idx_max = 50003      # To test for 1000 rows, use 51000
stepwise = True          # If true, also run iteratively of 1-delta CNN. This is time consuming.


# %% [code]
# Load CNN models, test file, and set up GA.# Load CNN models, test file, and set up GA.

myprev = {'cnn_static':0, 'cnn_dynamic':0, 'cnn_total':0, 'cnn_many':0,  'ga_total':0}
mytime = myprev.copy()

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
               crossover_rate=ga_cross, mutation_rate=ga_mutate,
               mut_div = ga_mut_div,
               tracking=track_details, save_states=ga_save_states)


# %% [code]
# Actual run

mylog('GA run started')
all_results = []
all_submissions = []
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
delta_groups = [list() for j in range(5)]
with open(output_dir + 'details.txt', 'w') as detail_file:
    for idx, row in data.iterrows():
        if idx < game_idx_min:
            continue
        if idx > game_idx_max:
            break
        delta = row[0]
        if not delta in deltaset:
            continue
        
        tf_arr = np.array(row[1:]).astype(np.float32).reshape((25, 25, 1))
        delta_groups[delta-1].append(tf_arr)
        if len(delta_groups[delta-1]) < batch_size:
            continue

        np.array(delta_groups[delta-1]).reshape(1, batch_size, 25, 25, 1)
        tic('cnn_total')
        solv_1 = cnn_manager.revert(delta_groups[delta-1], delta, 
                                    ga_static_1, ga_static_n, ga_dynamic_1, ga_dynamic_n)
#         toc('cnn_total')
#         tic('ga_total')
#         submission, res = ga.revert(idx, delta, delta_groups[delta-1].astype(bool), solv_1)
#         toc('ga_total')
#         all_submissions.append(submission)
#         all_results.append(res)
#         delta_groups[delta-1] = None
#         if track_details:
#             res_dict = dict(zip(result_header[:7], res[:7]))
#             detail_file.write('Details for {}:\n{}\n\n'.format(
#                 res_dict, ga.summary()))
#         if idx % status_freq == 0:
#             mylog('Completed game {}.'.format(idx))

# save_results(all_submissions, all_results, start_time, datetime.now().isoformat(' '))
mylog('Conway solver completed.')
