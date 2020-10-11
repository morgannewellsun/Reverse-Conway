# Fine tune the provided reverse game solutions by genetic algorithm

import pandas as pd
import random    # don't use np.random since it accepts only int32, too small.
from .conwaymap import ConwayMap


class ReverseGa:
    
    def __init__(self, conway:ConwayMap, pop_size, max_iters,
                 crossover_rate = 1, mutation_rate = 0.5, tracking = True):
        self.conway = conway
        self._chromo_len = conway.nrows * conway.ncols
        self._max_chromo = 2 ** self._chromo_len - 1
        self.pop_size = pop_size
        self._max_iters = max_iters
        self._max_muts = 4
        self._ncrossover = int(pop_size * crossover_rate / 2)
        self._nmutations = int(pop_size * mutation_rate)
        self._nimproves = int(pop_size * 0.5)
        self._tracking = tracking
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_colwidth', None)


    def revert(self, delta, stop_state, guess = set()):
        """ Arguments:
            stop_state is the integer representation of the stop state.
            guess is a set of initial states.
            Return:
            A tuple of two:
            initial state represented by a binary integer
            and the number of errors, which is the count of
            number of end cells from the initial state different from stop_state.
        """
        self._reset_all()
        self._delta = delta
        self._target = stop_state
        self._mutants = (guess | {0, stop_state})
        if len(guess) < self.pop_size:
            # Not enough intial guesses are supplied. Use random states.
            self._mutants |= {random.randint(3, self._max_chromo-5)
                              for j in range(self.pop_size)}
        self._select()
        self._track()      # Generation 0 is done.
        
        while self._gen_idx < self._max_iters:
            self._gen_idx += 1
            if self._best_error == 0:
                break
            self._reset_gen()
            self._mutate()
            self._crossover()
            self._improve()
            self._select()
            self._track()
        return self._curr_pop.iloc[0, 0], self._best_error


    def refine_cnn(self, game_idx, delta, target, cnn_result):
        # Arg target is the 1D 0/1 integer array representation of the end state.
        # Arg cnn_result is a 1D numpy array of probabilities as result of CNN.
        life50 = sum(cnn_result < 0.5)
        half_pop = int(self.pop_size / 2)
        selected = list(range(life50 - half_pop, life50 + half_pop))
        sorted_probs = sorted(cnn_result)
        # This is a list of 1D 0/1 arrays representing the boards from CNN.
        initial = [int(''.join(map(str, map(int, cnn_result > sorted_probs[j]))), 2) for j in selected]
        end_state = int(''.join(map(str, target)), 2)
        ga_result, ga_errors = self.revert(delta, end_state, guess = set(initial))
        
        if not self._tracking:
            return [game_idx, ga_result]
        
        target_lives = sum(target)
        cnn_guess = initial[half_pop]
        cnn_lives = bin(cnn_guess).count('1')
        cnn_errors = bin(self.conway.run(cnn_guess, self._delta)^self._target).count('1')
        ga_lives = bin(ga_result).count('1')
        return [game_idx, delta, target_lives, cnn_lives, cnn_errors,
                ga_lives, ga_errors, end_state, cnn_guess, ga_result]


    def _reset_gen(self):
        self._curr_pop.reset_index(drop=True, inplace=True)
        self._mutants = set()  # in each generation, chromos from mutation
        self._babies = set()   # in each generation, chromos from crossover
        self._improved = set() # calibrated from earlier iteration
        self._newpop = set()   # in each generation, new chromos not seen in past


    def _reset_all(self):
        self._curr_pop = pd.DataFrame(columns=('chromo', 'error', 'diff'))
        self._evaluated = set()    # all board evaluated before.
        self._gen_idx = 0
        self._best_gen = 0
        self._best_error = self._max_chromo
        if self._tracking:
            self._report = list()
        self._reset_gen()
        

    def _mutate(self):
        # rand_bits = random.choices(range(1, self._chromo_len), k=self._nmutations)
        rand_bits = [random.sample(range(self._chromo_len), n+1)
                     for j in range(self._nmutations)
                     for n in range(self._max_muts)]
        parents = random.choices(self._curr_pop['chromo'].tolist(),
                                 k = self._nmutations)
        self._mutants = {p ^ sum([1<<b for b in bb])
                         for p, bb in zip(parents, rand_bits)}


    def _crossover(self):
        parents = [random.sample(self._curr_pop['chromo'].tolist(), k=2)
                   for j in range(self._ncrossover)]
        masks = [random.randint(1, self._max_chromo-1)
                 for j in range(self._ncrossover)]
        masks = [(m, self._max_chromo^m) for m in masks]
        children = [[(p[0] & m[0]) | (p[1] & m[1]), (p[1] & m[0]) | (p[0] & m[1])]
                    for p, m in zip(parents, masks)]
        self._babies = {c for cc in children for c in cc}


    def _improve(self):
        selected = random.sample(range(self.pop_size), self._nimproves)
        self._improved = {row[0]^(~row[2]) for row in
                          self._curr_pop.iloc[selected,:].to_numpy()}


    def _select(self):
        """ Trim down current population to pop_size
        """
        self._newpop = (self._mutants | self._babies | self._improved) - self._evaluated
        diffs = [[b, bin(c).count('1'), c] for b in self._newpop 
                 for c in [self.conway.run(b, self._delta)^self._target]]
        self._evaluated |= self._newpop
        addition = pd.DataFrame(diffs, columns=('chromo', 'error', 'diff'))
        self._curr_pop = self._curr_pop.append(addition, ignore_index=True)
        self._curr_pop.sort_values('error', inplace=True)
        self._curr_pop = self._curr_pop.iloc[:self.pop_size]
        best = self._curr_pop.iloc[0, 1]
        if best < self._best_error:
            self._best_error = best
            self._best_gen = self._gen_idx


    def _track(self):
        if not self._tracking:
            return
        selected = set(self._curr_pop['chromo'])
        best_chromo = self._curr_pop.iloc[0, 0]
        src = ''
        if self._curr_pop.index[0] < self.pop_size:
            src += 'P'
        if best_chromo in self._mutants:
            src += 'M'
        if best_chromo in self._babies:
            src += 'B'
        if best_chromo in self._improved:
            src += 'I'
        self._report.append([
            len(self._mutants), len(self._babies),
            len(self._newpop), len(self._improved),
            len((self._mutants | self._babies | self._improved) & self._evaluated), # hits
            sum(self._curr_pop.index < self.pop_size),  # survivals
            len(selected & self._mutants),  # selected mutants
            len(selected & self._babies),   # selected babies
            len(selected & self._improved),
            len(self._evaluated), self._best_error,
            self._curr_pop.iloc[-1, 1], self._best_gen,
            src
            ])


    def summary(self):
        return pd.DataFrame(
            self._report, columns=(
                'mut', 'bab', 'new', 'imp', 'hit',
                'p_in', 'm_in', 'b_in', 'i_in',
                'eval', 'min_e', 'max_e', 'best', 'src'))