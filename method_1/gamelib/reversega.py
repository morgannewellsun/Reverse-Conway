# Fine tune the provided reverse game solutions by genetic algorithm

import pandas as pd
import random    # don't use np.random since it accepts only int32, too small.
import logging
from .gamelife import GameLife


class ReverseGa:
    
    def __init__(self, nrows, ncols, pop_size, max_iters,
                 crossover_rate = 1, mutation_rate = 0.5, tracking = True):
        self._gl = GameLife(nrows, ncols)
        self._chromo_len = nrows * ncols
        self._max_chromo = 2  ** self._chromo_len - 1
        self._pop_size = pop_size
        self._max_iters = max_iters
        self._ncrossover = int(pop_size * crossover_rate / 2)
        self._nmutations = int(pop_size * mutation_rate)
        self._tracking = tracking
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)


    def revert(self, delta, stop_state, guess):
        """ Arguments:
            stop_state is a flat list.
            guess is a tuple or list or set of initial states.
        """
        self._delta = delta
        self._target = self._gl.array_to_binary(stop_state)
        self._reset_all()

        self._mutants = {self._gl.array_to_binary(ss) for ss in guess}
        if len(guess) < self._pop_size:
            self._mutants |= {random.randint(3, self._max_chromo-5)
                              for j in range(self._pop_size)}
        self._select()
        
        while self._gen_idx < self._max_iters:
            self._gen_idx += 1
            self._reset_gen()
            self._mutate()
            self._crossover()
            self._select()
            if self._best_error == 0:
                break
        if self._tracking:
            self._report = pd.DataFrame(
                self._report, columns=(
                    'mut', 'bab', 'new', 'p_in', 'm_in', 'b_in',
                    'eval', 'min_e', 'max_e', 'best', 'src'))
            logging.info('\n{}'.format(self._report))
        return self._best_error


    def _reset_gen(self):
        self._mutants = set()  # in each generation, new chromos from mutation
        self._babies = set()   # in each generation, new chromos from crossover


    def _reset_all(self):
        self._reset_gen()
        self._curr_pop = pd.DataFrame(columns=('chromo', 'diffs'))
        self._evaluated = set()    # all board evaluated before.
        self._gen_idx = 0
        self._best_gen = 0
        self._best_error = self._max_chromo
        self._delta = 0
        if self._tracking:
            self._report = list()
        

    def _mutate(self):
        rand_bits = random.choices(range(1, self._chromo_len+1), k=self._nmutations)
        parents = random.choices(self._curr_pop['chromo'].tolist(), k = self._nmutations)
        self._mutants = {p^(1<<b) for p, b in zip(parents, rand_bits)}


    def _crossover(self):
        parents = [random.sample(self._curr_pop['chromo'].tolist(), k=2)
                   for j in range(self._ncrossover)]
        masks = [random.randint(1, self._max_chromo-1)
                 for j in range(self._ncrossover)]
        masks = [(m, self._max_chromo^m) for m in masks]
        children = [[(p[0] & m[0]) | (p[1] & m[1]), (p[1] & m[0]) | (p[0] & m[1])]
                    for p, m in zip(parents, masks)]
        self._babies = {c for cc in children for c in cc}


    def _select(self):
        """ Trim down current population to _pop_size
        """
        new_pop = (self._mutants | self._babies) - self._evaluated
        diffs = [[b, bin(self._gl.run_binary(b, self._delta)^self._target).count('1')]
                 for b in new_pop]
        self._evaluated |= new_pop
        addition = pd.DataFrame(diffs, columns=('chromo', 'diffs'))
        self._curr_pop = self._curr_pop.append(addition, ignore_index=True)
        self._curr_pop.sort_values('diffs', inplace=True)
        self._curr_pop = self._curr_pop.iloc[:self._pop_size]
        best = self._curr_pop.iloc[0, 1]
        if best < self._best_error:
            self._best_error = best
            self._best_gen = self._gen_idx
        if self._tracking:
            selected = set(self._curr_pop['chromo'])
            best_chromo = self._curr_pop.iloc[0, 0]
            src = ''
            if self._curr_pop.index[0] < self._pop_size:
                src += 'P'
            if best_chromo in self._mutants:
                src += 'M'
            if best_chromo in self._babies:
                src += 'B'
            self._report.append([
                len(self._mutants), len(self._babies), len(new_pop),
                sum(self._curr_pop.index < self._pop_size),  # survivals
                len(selected & self._mutants),  # selected mutants
                len(selected & self._babies),   # selected babies
                len(self._evaluated), self._best_error,
                self._curr_pop.iloc[-1, 1], self._best_gen,
                src
                ])
            self._curr_pop.reset_index(drop=True, inplace=True)

