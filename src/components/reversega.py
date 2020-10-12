# Fine tune the provided reverse game solutions by genetic algorithm

import numpy as np
import pandas as pd
from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn


class ReverseGa:
    
    def __init__(self, conway:BinaryConwayForwardPropFn,
                 pop_size = 10, max_iters = 10,
                 crossover_rate = 1, mutation_rate = 0.5,
                 mut_div = 10, tracking = True):
        # Arg mut_div: probability of mutation is 1/mut_div
        self.conway = conway
        self._chromo_len = conway.nrows * conway.ncols
        self.pop_size = pop_size
        self._max_iters = max_iters
        self._nmutations = int(pop_size * mutation_rate)
        self._mutation_div = mut_div
        self._ncrossover = int(pop_size * crossover_rate / 2)
        self._tracking = tracking
        self._offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]


    def revert(self, delta, stop_state, guess = None):
        """ Arguments:
            stop_state is the 4D array representation of the stop state,
            of shape (1, game_board_width, game_board_height, 1)
            guess is a 4D array of shape
            (batch_size, game_board_width, game_board_height, 1).
            Return:
            A tuple of two:
            initial state represented by a 4D array
            and the number of errors, which is the count of
            number of end cells from the initial state different from stop_state.
        """
        self._delta = delta
        self._target = stop_state
        self._reset(guess)
        self._select()
        self._track()      # Generation 0 is done.
        
        while self._gen_idx < self._max_iters:
            self._gen_idx += 1
            if self._best_error == 0:
                break
            self._mutate()
            self._crossover()
            self._select()
            self._track()
        return self._curr_pop[0]


    def refine_cnn(self, game_idx, delta, target, cnn_result):
        # Arg target is the 1D array of 0/1 integers representation of the end state.
        # Arg cnn_result is a 4D numpy array of probabilities as result of CNN,
        # of size (1, 25, 25, 1)
        if cnn_result is None:
            initial = None
            cnn_guess = np.array([0])
            cnn_lives = 0
            cnn_errors = 0
        else:
            life50 = (cnn_result < 0.5).sum()
            half_pop = int(self.pop_size / 2)
            selected = list(range(life50 - half_pop, life50 + half_pop))
            sorted_probs = sorted(cnn_result.flatten())
            # This is a list of 1D 0/1 arrays representing the boards from CNN.
            initial = [(cnn_result[0] > sorted_probs[j]).tolist() for j in selected]
            initial = np.array(initial).astype(int)
            cnn_guess = initial[half_pop]
            cnn_lives = cnn_guess.sum()
            cnn_errors = abs(self.conway(cnn_guess, delta) - target).sum()
        
        end_state = target.reshape(
            (1, self.conway.nrows, self.conway.ncols, 1))
        ga_result = self.revert(delta, end_state, guess = initial)
        if not self._tracking:
            return [game_idx, ga_result]
        
        target_lives = target.sum()
        ga_lives = ga_result.sum()
        return [game_idx, delta, target_lives, cnn_lives, cnn_errors,
                ga_lives, self._best_error, 
                ''.join(map(str, end_state.flatten().tolist())),
                ''.join(map(str, cnn_guess.flatten().tolist())),
                ''.join(map(str, ga_result.flatten().tolist())) ]


    def _reset(self, guess):
        # set up generation 0.
        self._curr_pop = None      # 4D np.array
        self._gen_idx = 0                # Current generation index.
        self._best_gen = 0               # The generate giving the best chromo
        self._best_error = self._chromo_len     # The smallest error so far
        if self._tracking:
            self._report = list()

        # Generation 0 start from building self._mutants + _babies.
        empty_state = np.array([0] * self._chromo_len).reshape(self._target.shape)
        self._mutants = np.concatenate((empty_state, self._target))
        if guess is None:
            # Not enough intial guesses are supplied. Use random states.
            sz = (self.pop_size, self.conway.nrows, self.conway.ncols, 1)
            self._babies = np.random.randint(2, size=sz)
        else:
            self._babies = guess


    def _mutate(self):
        c = np.random.choice(len(self._curr_pop), replace=False, size=self._nmutations)
        chromos = self._curr_pop[c]
        # The resulting board has 1 / self._mutation_div fraction being live cells.
        muter = (np.random.randint(self._mutation_div, size=chromos.shape)
                 / (self._mutation_div - 1)).astype(int)
        # Outside this area, we don't mutate any cells.
        target_area = sum([np.roll(self._diffs[c], shift, (-3, -2)) for shift in self._offsets])
        muter *= target_area
        self._mutants = (chromos + muter) % 2
        # In case muter[j] are all zeros, no mutation happens. Remove it.
        self._mutants = self._mutants[self._mutants.sum(axis=(1,2,3))>0]


    def _crossover(self):
        idx = np.random.choice(len(self._curr_pop), size=2*self._ncrossover, replace=True)
        dads = self._curr_pop[idx[:self._ncrossover]]
        moms = self._curr_pop[idx[self._ncrossover:]]
        swapper = np.random.randint(low=0, high=2, size=(
            self._ncrossover, self.conway.nrows, self.conway.ncols, 1))
        complim = 1 - swapper
        self._babies = np.concatenate((
            dads * swapper + moms * complim,
            dads * complim + moms * swapper ))
        self._babies = self._babies[self._babies.sum(axis=(1,2,3))>0]


    def _add_newpop(self, newpop):
        # Add newpop into the current population.
        # Arg newpop must be 4D np.array.
        # The new population after Conway game forward is compared to the target.
        # The differences and the number of errors are recorded.

        # self._diffs is 4D np.array of differences from target
        # self._errors is 1D array of error count.

        newdiff = abs(self.conway(newpop)
                      - np.repeat(self._target, len(newpop), axis=0))
        newerr = np.array(newdiff.sum(axis=(1,2,3)))
        if self._curr_pop is None:
            self._curr_pop = newpop
            self._diffs = newdiff
            self._errors = newerr
        else:
            self._curr_pop = np.concatenate((self._curr_pop, newpop))
            self._diffs = np.concatenate((self._diffs, newdiff))
            self._errors = np.concatenate((self._errors, newerr))


    def _select(self):
        """ Trim down current population to pop_size
        """
        # The order of adding mutants and babies is important.
        # Later the statistics are computed assuming this addition order.
        self._add_newpop(self._mutants)
        self._add_newpop(self._babies)
        # Make sure the best and the worst chromos
        # after the selection are correctly placed.
        self._selects = np.argpartition(self._errors, (1, self.pop_size))
        self._selects = self._selects[0:self.pop_size]
        self._curr_pop = self._curr_pop[self._selects]
        self._diffs = self._diffs[self._selects]
        self._errors = self._errors[self._selects]
        best = self._errors[0]
        if best < self._best_error:
            self._best_error = best
            self._best_gen = self._gen_idx


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
            sum(self._selects < n0),  # survivals
            sum((self._selects >= n0) & (self._selects < n1)),  # selected mutants
            sum(self._selects >= n1),   # selected babies (from crossovers)
            self._best_error, self._errors[-1], self._best_gen, src
            ])


    def summary(self):
        return pd.DataFrame(
            self._report, columns=(
                'mut', 'bab', 'p_in', 'm_in', 'b_in',
                'min_e', 'max_e', 'best', 'src'))
