# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import pprint
from typing import *

import numpy as np
import pandas as pd
import tensorflow as tf
import time

# %% [code]
# =====================================================================================================================
# USER SETTINGS# USER SETTINGS

EVALUATE = True
SUBMIT = False

MODEL_ROOT_DIR = '../input/conway1/'
CNN_PATHS = {
    # "delta_1_final": MODEL_ROOT_DIR + "delta_1_final",
    "delta_1": MODEL_ROOT_DIR + "delta_1",
    # "delta_2_final": MODEL_ROOT_DIR + "delta_2_final",
    # "delta_2": MODEL_ROOT_DIR + "delta_2",
    # "delta_3_final": MODEL_ROOT_DIR + "delta_3_final",
    # "delta_3": MODEL_ROOT_DIR + "delta_3",
    # "delta_4_final": MODEL_ROOT_DIR + "delta_4_final",
    "delta_4": MODEL_ROOT_DIR + "delta_4",
    # "delta_5_final": MODEL_ROOT_DIR + "delta_5_final",
    "delta_5": MODEL_ROOT_DIR + "delta_5"
}
KAGGLE_TEST_FILE_PATH = '../input/conways-reverse-game-of-life-2020/test.csv'
OUTPUT_DIR = './'

# Batch size affects both the CNN and the GA.
BATCH_SIZE = 256  # batch size

# The following settings control the CNN.
CNN_USE_DELTA_1_FINAL = False  # whether or not to use an aggressive model for delta = 1
CNN_USE_DELTA_2_FINAL = False  # whether or not to use an aggressive model for delta = 2
CNN_USE_DELTA_3_FINAL = False  # whether or not to use an aggressive model for delta = 3
CNN_USE_DELTA_4_FINAL = False  # whether or not to use an aggressive model for delta = 4
CNN_USE_DELTA_5_FINAL = False  # whether or not to use an aggressive model for delta = 5

# The following settings control genetic algorithm population initialization.
GA_STATIC_PROB_THRESHOLD_SPAN = 0.4  # half-width of range of thresholds for static
GA_STATIC_POP = 12  # GA initial population from the static prob
GA_DYNAMIC_POP = 50  # GA initial population from the dynamic prob

# The following settings control genetic algorithm dynamics.
GA_SMART = True
GA_ITERATIONS = {
    1: 15,
    2: 20,
    3: 20,
    4: 80,
    5: 80,
}
GA_MUTATION_TYPE = "MUTATION_RATE"
GA_MUTATION_RATE = 0.006
GA_MUTATIONS_PER_BOARD = 5  # unused
GA_MUTATION_RATES_MULTIPLE = (0.05, 0.1, 0.2)  # unused

# The following settings restricts to only a selected subset of data to test.
DELTA_SET = {1, 2, 3, 4, 5}  # Load only the model for specified deltas. To load all, use {1,2,3,4,5}
GAME_IDX_MIN = 50000  # Kaggle test game indices from 50000 to 99999.
GAME_IDX_MAX = 55000  # To test for 1000 rows, use 51000

# Don't touch.
GA_TOTAL_POP = GA_STATIC_POP + GA_DYNAMIC_POP + 2
if SUBMIT:
    for delta in [1, 2, 3, 4, 5]:
        assert delta in DELTA_SET
    assert GAME_IDX_MIN == 50000
    assert GAME_IDX_MAX == 99999


# %% [code]
# =====================================================================================================================
# Conway's game logic


class BinaryConwayForwardPropFn:

    def __init__(self, numpy_mode=False, nrows=25, ncols=25):
        self._numpy_mode = numpy_mode
        self.nrows = nrows
        self.ncols = ncols
        self._moore_offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i != 0 or j != 0)]
        if not numpy_mode:
            self._moore_offsets = [tf.constant(tup, dtype=tf.dtypes.int32) for tup in self._moore_offsets]

    def __call__(self, inputs, delta=1):
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
            live_neighbor_counts = tf.math.count_nonzero(neighbors, axis=0, dtype=tf.int32)
            two_live_neighbors = tf.math.equal(live_neighbor_counts, 2)
            three_live_neighbors = tf.math.equal(live_neighbor_counts, 3)
            outputs = tf.math.logical_or(three_live_neighbors, tf.math.logical_and(two_live_neighbors, inputs))
        return outputs


# %% [code]
# =====================================================================================================================
# Generate initial reverse same guesses using CNN.


class CNN:
    # Initializes a GA population using a CNN reverse model.

    def __init__(
            self,
            *,
            model: tf.function,
            population_size_static: int,
            population_size_dynamic: int,
            static_prob_threshold_span: float):
        # Arg cnn_reverters is a dictionary from int (delta)
        # to a CNN solver. A solver accepts array of np.float32.
        self._model = model
        self._population_size_static = population_size_static
        self._population_size_dynamic = population_size_dynamic
        self._bars_static = tf.reshape(
            (
                tf.linspace(
                    0.5 - static_prob_threshold_span,
                    0.5 + static_prob_threshold_span,
                    num=self._population_size_static)
                if self._population_size_static > 1
                else tf.constant(0.5)
            ),
            shape=(-1, 1, 1, 1, 1))

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(BATCH_SIZE, 25, 25, 1), dtype=tf.bool)])
    def _revert_static(self, stop_state) -> tf.Tensor:
        cnn_result = self._model(tf.cast(stop_state, tf.float32))  # (batch, 25, 25, 1)
        x = tf.greater_equal(cnn_result, self._bars_static)  # (pop, batch, 25, 25, 1)
        return x

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(BATCH_SIZE, 25, 25, 1), dtype=tf.bool)])
    def _revert_dynamic(self, stop_state) -> tf.Tensor:
        cnn_result = self._model(tf.cast(stop_state, tf.float32))  # (batch, 25, 25, 1)
        noise = tf.random.uniform(
            shape=(self._population_size_dynamic, 1, 25, 25, 1),
            minval=0, maxval=1,
            dtype=tf.dtypes.float32)
        x = tf.greater_equal(cnn_result, noise)  # (pop, batch, 25, 25, 1)
        return x

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(BATCH_SIZE, 25, 25, 1), dtype=tf.bool)])
    def revert(self, stop_state: tf.Tensor) -> tf.Tensor:
        # stop_state: (batch, 25, 25, 1)
        # output: (population, batch, 25, 25, 1)
        stop_state_expanded = tf.expand_dims(stop_state, axis=0)
        all_zeros = tf.zeros_like(stop_state_expanded, dtype=tf.bool)
        guesses = [stop_state_expanded, all_zeros]
        if self._population_size_dynamic != 0:
            guesses.append(self._revert_dynamic(stop_state))
        if self._population_size_static != 0:
            guesses.append(self._revert_static(stop_state))
        return tf.concat(guesses, axis=0)


# %% [code]
# =====================================================================================================================
# Enhance revert game results using genetic algorithm.


class GA:

    def __init__(self, *, iterations: int, delta: int):
        self._conway = BinaryConwayForwardPropFn()
        self._iterations = iterations
        self._delta = delta
        if GA_MUTATION_TYPE == "MUTATIONS_PER_BOARD":
            self._mutations_per_board = tf.constant(GA_MUTATIONS_PER_BOARD)
        elif GA_MUTATION_TYPE == "MUTATION_RATE":
            self._mutation_rate = tf.constant(GA_MUTATION_RATE, dtype=tf.float32)
        elif GA_MUTATION_TYPE == "MUTATION_RATES_MULTIPLE":
            self._mutation_rates_multiple = tf.reshape(
                tf.constant(GA_MUTATION_RATES_MULTIPLE, dtype=tf.float32),
                shape=(len(GA_MUTATION_RATES_MULTIPLE), 1, 1, 1, 1, 1))
        else:
            raise NotImplementedError
        self._six_twenty_five = tf.constant(625, dtype=tf.int32)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(GA_TOTAL_POP, BATCH_SIZE, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, BATCH_SIZE, 25, 25, 1), dtype=tf.bool)])
    def _generate_mutations(self, population: tf.Tensor, population_diffs: tf.Tensor) -> tf.Tensor:
        target_areas = tf.reduce_any(
            [tf.roll(population_diffs, (i, j), (-3, -2)) for i in [-1, 0, 1] for j in [-1, 0, 1]],
            axis=0)  # (pop, batch, 25, 25, 1)
        target_area_sizes = tf.reshape(
            tf.math.count_nonzero(target_areas, axis=(-3, -2, -1), dtype=tf.int32),
            shape=(GA_TOTAL_POP, -1, 1, 1, 1))  # (pop, batch, 1, 1, 1)
        if GA_MUTATION_TYPE == "MUTATIONS_PER_BOARD":
            muter_threshold = tf.cast(
                self._mutations_per_board / (target_area_sizes + 1),
                dtype=tf.float32)  # (pop, batch, 1, 1, 1)
        elif GA_MUTATION_TYPE == "MUTATION_RATE":
            muter_threshold = self._mutation_rate
        elif GA_MUTATION_TYPE == "MUTATION_RATES_MULTIPLE":
            muter_threshold = self._mutation_rates_multiple
        else:
            raise NotImplementedError
        muters = tf.random.uniform(
            shape=(GA_TOTAL_POP, 1, 25, 25, 1),
            minval=0, maxval=1,
            dtype=tf.dtypes.float32)  # (pop, 1, 25, 25, 1)
        muters = tf.greater_equal(muter_threshold, muters)  # (pop, batch, 25, 25, 1)
        muters &= target_areas  # (pop, batch, 25, 25, 1)
        mutants = tf.math.logical_xor(population, muters)
        if GA_MUTATION_TYPE == "MUTATION_RATES_MULTIPLE":
            return tf.reshape(mutants, shape=(len(GA_MUTATION_RATES_MULTIPLE) * GA_TOTAL_POP, BATCH_SIZE, 25, 25, 1))
        else:
            return mutants

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(GA_TOTAL_POP, BATCH_SIZE, 25, 25, 1), dtype=tf.bool)])
    def _generate_crossovers(self, population: tf.Tensor) -> tf.Tensor:
        shuffled_population = tf.random.shuffle(population)
        cell_swapper = tf.greater_equal(
            tf.random.uniform(shape=(GA_TOTAL_POP, 1, 25, 25, 1), minval=0, maxval=1),
            0.5)
        cell_complement = tf.logical_not(cell_swapper)
        return tf.logical_or(
            tf.logical_and(population, cell_swapper),
            tf.logical_and(shuffled_population, cell_complement))

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(BATCH_SIZE, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, BATCH_SIZE, 25, 25, 1), dtype=tf.bool)])
    def _calculate_diffs_and_match_counts(
            self, target: tf.Tensor, population: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        population = self._conway(population, delta=self._delta)
        diffs = tf.math.logical_xor(population, target)  # (pop, batch, 25, 25, 1)
        match_counts = (
                self._six_twenty_five
                - tf.transpose(tf.math.count_nonzero(diffs, axis=(-3, -2, -1), dtype=tf.int32)))  # (batch, pop)
        return diffs, match_counts

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(BATCH_SIZE, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, BATCH_SIZE, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, BATCH_SIZE, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(BATCH_SIZE, GA_TOTAL_POP), dtype=tf.int32)])
    def _iterate(
            self,
            target: tf.Tensor,
            population: tf.Tensor,
            population_diffs: tf.Tensor,
            population_match_counts: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        mutations = self._generate_mutations(population, population_diffs)  # (pop_m, batch, 25, 25, 1)
        crossovers = self._generate_crossovers(population)  # (pop_c, batch, 25, 25, 1)
        mutation_diffs, mutation_diff_counts = self._calculate_diffs_and_match_counts(
            target, mutations)  # (pop_m, batch, 25, 25, 1) (batch, pop_m)
        crossover_diffs, crossover_diff_counts = self._calculate_diffs_and_match_counts(
            target, crossovers)  # (pop_c, batch, 25, 25, 1) (batch, pop_c)
        population = tf.concat(
            [population, mutations, crossovers], axis=0)  # (pop_pmc, batch, 25, 25, 1)
        population_diffs = tf.concat(
            [population_diffs, mutation_diffs, crossover_diffs], axis=0)  # (pop_pmc, batch, 25, 25, 1)
        population_match_counts = tf.concat(
            [population_match_counts, mutation_diff_counts, crossover_diff_counts], axis=1)  # (batch, pop_pmc)
        population_match_counts, best_indices = tf.math.top_k(
            population_match_counts, k=GA_TOTAL_POP, sorted=False)  # (batch, pop_p)
        population = tf.transpose(tf.gather(
            params=tf.transpose(population, perm=[1, 0, 2, 3, 4]),  # (batch, pop_p, 25, 25, 1)
            indices=best_indices,  # (batch, pop_p)
            axis=1,
            batch_dims=1), perm=[1, 0, 2, 3, 4])  # (pop_p, batch, 25, 25, 1)
        population_diffs = tf.transpose(tf.gather(
            params=tf.transpose(population_diffs, perm=[1, 0, 2, 3, 4]),  # (batch, pop_p, 25, 25, 1)
            indices=best_indices,  # (batch, pop_p)
            axis=1,
            batch_dims=1), perm=[1, 0, 2, 3, 4])  # (pop_p, batch, 25, 25, 1)
        return population, population_diffs, population_match_counts

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(BATCH_SIZE, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, BATCH_SIZE, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, BATCH_SIZE, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(BATCH_SIZE, GA_TOTAL_POP), dtype=tf.int32)])
    def _iterate_last(
            self,
            target: tf.Tensor,
            population: tf.Tensor,
            population_diffs: tf.Tensor,
            population_match_counts: tf.Tensor
    ) -> tf.Tensor:
        mutations = self._generate_mutations(population, population_diffs)  # (pop_m, batch, 25, 25, 1)
        crossovers = self._generate_crossovers(population)  # (pop_c, batch, 25, 25, 1)
        mutation_diffs, mutation_diff_counts = self._calculate_diffs_and_match_counts(
            target, mutations)  # (pop_m, batch, 25, 25, 1) (batch, pop_m)
        crossover_diffs, crossover_diff_counts = self._calculate_diffs_and_match_counts(
            target, crossovers)  # (pop_c, batch, 25, 25, 1) (batch, pop_c)
        population = tf.concat(
            [population, mutations, crossovers], axis=0)  # (pop_pmc, batch, 25, 25, 1)
        population_match_counts = tf.concat(
            [population_match_counts, mutation_diff_counts, crossover_diff_counts], axis=1)  # (batch, pop_pmc)
        population_match_counts, best_indices = tf.math.top_k(population_match_counts, k=1, sorted=False)  # (batch, 1)
        population = tf.transpose(tf.gather(
            params=tf.transpose(population, perm=[1, 0, 2, 3, 4]),  # (batch, 1, 25, 25, 1)
            indices=best_indices,  # (batch, pop_p)
            axis=1,
            batch_dims=1), perm=[1, 0, 2, 3, 4])  # (1, batch, 25, 25, 1)
        population = tf.squeeze(population, axis=0)
        return population

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(BATCH_SIZE, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, BATCH_SIZE, 25, 25, 1), dtype=tf.bool)])
    def refine(self, target: tf.Tensor, population: tf.Tensor) -> tf.Tensor:
        population_diffs, population_match_counts = self._calculate_diffs_and_match_counts(target, population)
        for _ in range(self._iterations - 1):
            population, population_diffs, population_match_counts = self._iterate(
                target, population, population_diffs, population_match_counts)
        return self._iterate_last(target, population, population_diffs, population_match_counts)

    def smart_refine_all_batches(
            self, target_batches: List[tf.Tensor], population_batches: List[tf.Tensor]
    ) -> List[tf.Tensor]:
        assert len(target_batches) == len(population_batches)
        n_batches = len(target_batches)
        batch_diffs = []
        batch_match_counts = []
        batch_match_count_sums = []
        for target_batch, population_batch in zip(target_batches, population_batches):
            ds, mcs = self._calculate_diffs_and_match_counts(target_batch, population_batch)
            batch_diffs.append(ds)
            batch_match_counts.append(mcs)
            batch_match_count_sums.append(tf.reduce_sum(mcs).numpy())
        np_batch_match_count_sums = np.array(batch_match_count_sums)
        batch_iterations = self._iterations * n_batches
        while batch_iterations > n_batches:
            probs = 1 / np_batch_match_count_sums
            probs = probs / np.sum(probs)
            idx = np.random.choice(range(len(probs)), p=probs)
            # print(np_batch_match_count_sums)
            # print(min_idx)
            population_batches[idx], batch_diffs[idx], batch_match_counts[idx] = self._iterate(
                target_batches[idx], population_batches[idx], batch_diffs[idx], batch_match_counts[idx])
            np_batch_match_count_sums[idx] = tf.reduce_sum(batch_match_counts[idx]).numpy()
            batch_iterations -= 1
        output_batches = []
        for t, p, d, mc in zip(target_batches, population_batches, batch_diffs, batch_match_counts):
            output_batches.append(self._iterate_last(t, p, d, mc))
        return output_batches


# %% [code]
# =====================================================================================================================
# Solver strategy definitions.


class Solver:

    def __init__(self, strategy: str, cnn_models: Dict[str, tf.function]):
        self._strategy = strategy
        self._cnn_models = cnn_models

    @staticmethod
    def _collect_into_batches(
            target_stop_states: List[np.ndarray], batch_size: int, zero_pad: bool = False
    ) -> Union[Tuple[List[np.ndarray], int], List[np.ndarray]]:
        batches = []
        batch_list = []
        for target_stop_state in target_stop_states:
            batch_list.append(target_stop_state)
            if len(batch_list) == batch_size:
                batches.append(np.stack(batch_list))
                batch_list = []
        last_pad_count = 0
        if len(batch_list) > 0:
            if zero_pad:
                while len(batch_list) < batch_size:
                    batch_list.append(np.zeros_like(batch_list[0]))
                    last_pad_count += 1
            batches.append(np.stack(batch_list))
        # print("Batches\n", len(batches), "\n", batches[0].shape, "\n", batches[-1].shape)
        # print(f"last_pad_count = {last_pad_count}")
        print(f"Data split into batches: {time.time() - MAIN_START_TIME}")
        if zero_pad:
            return batches, last_pad_count
        else:
            return batches

    def solve(self, delta: int, target_stop_states: List[np.ndarray]) -> List[np.ndarray]:
        if self._strategy == "stackwise":
            return self._solve_stackwise(delta, target_stop_states)
        elif self._strategy == "oneshot":
            return self._solve_oneshot(delta, target_stop_states)
        elif self._strategy == "all_models":
            return self._solve_all_models(delta, target_stop_states)
        else:
            raise NotImplementedError

    def _solve_stackwise(self, delta: int, target_stop_states: List[np.ndarray]) -> List[np.ndarray]:
        # target_stop_states: [(25, 25, 1)]
        # returns: [(25, 25, 1)]

        batches, last_pad_count = Solver._collect_into_batches(
            target_stop_states, BATCH_SIZE, zero_pad=True)  # [(batch, 25, 25, 1)]
        target_stop_batches = [tf.constant(batch, dtype=tf.bool) for batch in batches]
        batches = [tf.constant(batch, dtype=tf.bool) for batch in batches]
        iterations = GA_ITERATIONS[delta]

        cnn_reverter = None

        steps_back = 0
        while delta > 1:
            if cnn_reverter is None:
                cnn_reverter = CNN(
                    model=self._cnn_models[f"delta_1"],
                    population_size_static=GA_STATIC_POP,
                    population_size_dynamic=GA_DYNAMIC_POP,
                    static_prob_threshold_span=GA_STATIC_PROB_THRESHOLD_SPAN)
            cnn_output_batches = []  # [(pop, batch, 25, 25, 1)]
            for batch in batches:
                out = cnn_reverter.revert(batch)
                # print("CNN\n", out.get_shape())
                cnn_output_batches.append(out)
            print(f"CNN used to revert from delta {delta} to {delta - 1}: {time.time() - MAIN_START_TIME}")
            steps_back += 1
            ga_refiner = GA(iterations=iterations, delta=steps_back)
            if GA_SMART:
                ga_output_batches = ga_refiner.smart_refine_all_batches(target_stop_batches, cnn_output_batches)
            else:
                ga_output_batches = []
                for target_stop_batch, cnn_output_batch in zip(target_stop_batches, cnn_output_batches):
                    out = ga_refiner.refine(target_stop_batch, cnn_output_batch)
                    # print("GA\n", out.get_shape())
                    ga_output_batches.append(out)
            print(f"GA used to refine CNN predictions for delta {delta - 1}: {time.time() - MAIN_START_TIME}")
            batches = ga_output_batches
            delta -= 1

        cnn_reverter = CNN(
            model=self._cnn_models["delta_1_final" if CNN_USE_DELTA_1_FINAL else "delta_1"],
            population_size_static=GA_STATIC_POP,
            population_size_dynamic=GA_DYNAMIC_POP,
            static_prob_threshold_span=GA_STATIC_PROB_THRESHOLD_SPAN)
        cnn_output_batches = []  # [(pop, batch, 25, 25, 1)]
        for batch in batches:
            cnn_output_batches.append(cnn_reverter.revert(batch))
        print(f"CNN used to revert from delta 1 to predicted start: {time.time() - MAIN_START_TIME}")
        steps_back += 1
        ga_refiner = GA(iterations=iterations, delta=steps_back)
        if GA_SMART:
            output_batches = ga_refiner.smart_refine_all_batches(target_stop_batches, cnn_output_batches)
        else:
            output_batches = []
            for target_stop_batch, cnn_output_batch in zip(target_stop_batches, cnn_output_batches):
                output_batches.append(ga_refiner.refine(target_stop_batch, cnn_output_batch))
        print(f"GA used to refine CNN predictions for predicted start: {time.time() - MAIN_START_TIME}")

        output = []
        for output_batch in output_batches:
            output.extend(list(output_batch.numpy()))
        return output[: -1 * last_pad_count] if last_pad_count > 0 else output

    def _solve_oneshot(self, delta: int, target_stop_states: List[np.ndarray]) -> List[np.ndarray]:
        # target_stop_states: [(25, 25, 1)]
        # returns: [(25, 25, 1)]

        batches, last_pad_count = Solver._collect_into_batches(
            target_stop_states, BATCH_SIZE, zero_pad=True)  # [(batch, 25, 25, 1)]
        batches = [tf.constant(batch, dtype=bool) for batch in batches]

        model_name = f"delta_{delta}"
        if (
                (delta == 1 and CNN_USE_DELTA_1_FINAL)
                or (delta == 2 and CNN_USE_DELTA_2_FINAL)
                or (delta == 3 and CNN_USE_DELTA_3_FINAL)
                or (delta == 4 and CNN_USE_DELTA_4_FINAL)
                or (delta == 5 and CNN_USE_DELTA_5_FINAL)):
            model_name += "_final"
        model = self._cnn_models[model_name]

        cnn_reverter = CNN(
            model=model,
            population_size_static=GA_STATIC_POP,
            population_size_dynamic=GA_DYNAMIC_POP,
            static_prob_threshold_span=GA_STATIC_PROB_THRESHOLD_SPAN)
        ga_refiner = GA(iterations=GA_ITERATIONS[delta], delta=delta)

        cnn_output_batches = []
        for batch in batches:
            cnn_output_batches.append(cnn_reverter.revert(batch))
        print(f"CNN used to revert from delta {delta} to predicted start: {time.time() - MAIN_START_TIME}")
        if GA_SMART:
            ga_output_batches = ga_refiner.smart_refine_all_batches(batches, cnn_output_batches)
        else:
            ga_output_batches = []
            for batch, cnn_output_batch in zip(batches, cnn_output_batches):
                ga_output_batches.append(ga_refiner.refine(batch, cnn_output_batch))
        print(f"GA used to refine CNN predictions for predicted start: {time.time() - MAIN_START_TIME}")

        output = []
        for ga_output_batch in ga_output_batches:
            output.extend(list(ga_output_batch.numpy()))
        return output[: -1 * last_pad_count] if last_pad_count > 0 else output

    def _solve_all_models(self, delta: int, target_stop_states: List[np.ndarray]) -> List[np.ndarray]:
        # target_stop_states: [(25, 25, 1)]
        # returns: [(25, 25, 1)]

        batches, last_pad_count = Solver._collect_into_batches(
            target_stop_states, BATCH_SIZE, zero_pad=True)  # [(batch, 25, 25, 1)]
        batches = [tf.constant(batch, dtype=bool) for batch in batches]

        models = []
        model_name = f"delta_{delta}"
        models.append(self._cnn_models[model_name])
        if (
                (delta == 1 and CNN_USE_DELTA_1_FINAL)
                or (delta == 2 and CNN_USE_DELTA_2_FINAL)
                or (delta == 3 and CNN_USE_DELTA_3_FINAL)
                or (delta == 4 and CNN_USE_DELTA_4_FINAL)
                or (delta == 5 and CNN_USE_DELTA_5_FINAL)):
            model_name += "_final"
        models.append(self._cnn_models[model_name])

        cnn_reverters = (
            [
                CNN(
                    model=models[0],
                    population_size_static=GA_STATIC_POP,
                    population_size_dynamic=GA_DYNAMIC_POP,
                    static_prob_threshold_span=GA_STATIC_PROB_THRESHOLD_SPAN)
            ]
            if len(models) == 1 else
            [
                CNN(
                    model=models[0],
                    population_size_static=GA_STATIC_POP // 2,
                    population_size_dynamic=GA_DYNAMIC_POP // 2,
                    static_prob_threshold_span=GA_STATIC_PROB_THRESHOLD_SPAN),
                CNN(
                    model=models[1],
                    population_size_static=GA_STATIC_POP - (GA_STATIC_POP // 2),
                    population_size_dynamic=GA_DYNAMIC_POP - (GA_DYNAMIC_POP // 2) - 2,
                    static_prob_threshold_span=GA_STATIC_PROB_THRESHOLD_SPAN)
            ])
        ga_refiner = GA(iterations=GA_ITERATIONS[delta], delta=delta)

        cnn_output_batches = []
        for batch in batches:
            cnn_output_batch_list = []
            for cnn_reverter in cnn_reverters:
                cnn_output_batch_list.append(cnn_reverter.revert(batch))
            cnn_output_batches.append(tf.concat(cnn_output_batch_list, axis=0))
        print(f"CNN used to revert from delta {delta} to predicted start: {time.time() - MAIN_START_TIME}")
        if GA_SMART:
            ga_output_batches = ga_refiner.smart_refine_all_batches(batches, cnn_output_batches)
        else:
            ga_output_batches = []
            for batch, cnn_output_batch in zip(batches, cnn_output_batches):
                ga_output_batches.append(ga_refiner.refine(batch, cnn_output_batch))
        print(f"GA used to refine CNN predictions for predicted start: {time.time() - MAIN_START_TIME}")

        output = []
        for ga_output_batch in ga_output_batches:
            output.extend(list(ga_output_batch.numpy()))
        return output[: -1 * last_pad_count] if last_pad_count > 0 else output


# %% [code]
# =====================================================================================================================
# Main function.


class Evaluator:
    conway = BinaryConwayForwardPropFn(numpy_mode=True)

    @staticmethod
    def evaluate(delta: int, predicted_start_states: List[np.ndarray], target_stop_states: List[np.ndarray]) -> Dict:
        target_stop_live_cell_count = 0
        predicted_stop_live_cell_count = 0
        predicted_start_live_cell_count = 0
        stop_error_cell_count = 0
        total_cell_count = 0
        for predicted_start_state, target_stop_state in zip(predicted_start_states, target_stop_states):
            target_stop_live_cell_count += np.sum(target_stop_state)
            predicted_start_live_cell_count += np.sum(predicted_start_state)
            predicted_stop_state = predicted_start_state
            for _ in range(delta):
                predicted_stop_state = Evaluator.conway(predicted_stop_state)
            predicted_stop_live_cell_count += np.sum(predicted_stop_state)
            stop_error_cell_count += 625 - np.count_nonzero(np.equal(predicted_stop_state, target_stop_state))
            total_cell_count += 625
        stats = {
            # "target_stop_live_cell_count": target_stop_live_cell_count,
            # "predicted_stop_live_cell_count": predicted_stop_live_cell_count,
            # "predicted_start_live_cell_count": predicted_start_live_cell_count,
            # "stop_error_cell_count": stop_error_cell_count,
            # "total_cell_count": total_cell_count,
            "target_stop_live_rate": target_stop_live_cell_count / total_cell_count,
            "predicted_stop_live_rate": predicted_stop_live_cell_count / total_cell_count,
            "predicted_start_live_rate": predicted_start_live_cell_count / total_cell_count,
            "error_rate": stop_error_cell_count / total_cell_count}
        return stats

    @staticmethod
    def submit(delta_indexes: Dict, delta_predicted_start_states: Dict):
        submissions = []
        for d in delta_predicted_start_states.keys():
            for i, s in zip(delta_indexes[d], delta_predicted_start_states[d]):
                submissions.append([i, *(s.flatten().astype(int).tolist())])
        cols = ['id']
        cols.extend(['start_' + str(j) for j in range(625)])
        df = pd.DataFrame(submissions, columns=cols)
        df = df.sort_values(by='id')
        df.to_csv(OUTPUT_DIR + 'submission.csv', index=False)


# %% [code]
# =====================================================================================================================
# Main function.

MAIN_START_TIME = None

if __name__ == "__main__":

    MAIN_START_TIME = time.time()
    print("")

    cnn_models = dict([(name, tf.saved_model.load(path)) for name, path in CNN_PATHS.items()])
    print(f"CNN models loaded: {time.time() - MAIN_START_TIME}")
    load_time = time.time() - MAIN_START_TIME

    data = pd.read_csv(KAGGLE_TEST_FILE_PATH, index_col=0, dtype='int')
    print(f"Data loaded: {time.time() - MAIN_START_TIME}")

    delta_groups = {1: [], 2: [], 3: [], 4: [], 5: []}
    delta_indexes = {1: [], 2: [], 3: [], 4: [], 5: []}
    delta_predicted_start_states = dict()
    for idx, row in data.iterrows():
        if idx < GAME_IDX_MIN:
            continue
        if idx > GAME_IDX_MAX:
            break
        delta = row[0]
        if delta not in DELTA_SET:
            continue
        target_stop_state = np.array(row[1:]).astype(np.float32).reshape((25, 25, 1))
        delta_groups[delta].append(target_stop_state)
        delta_indexes[delta].append(idx)
    print(f"Data parsed to numpy: {time.time() - MAIN_START_TIME}")

    delta_stats = {}

    solver = Solver(strategy="stackwise", cnn_models=cnn_models)
    for delta in [1, 2, 3]:
        target_stop_states = delta_groups[delta]
        if len(target_stop_states) == 0:
            continue
        print("")
        print(f"Starting to solve delta {delta}: {time.time() - MAIN_START_TIME}")
        predicted_start_states = solver.solve(delta, target_stop_states)
        delta_predicted_start_states[delta] = predicted_start_states
        if EVALUATE:
            print(f"Evaluating delta {delta}: {time.time() - MAIN_START_TIME}")
            stats = Evaluator.evaluate(delta, predicted_start_states, target_stop_states)
            pprint.pprint(stats, indent=4)
            delta_stats[delta] = stats

    solver = Solver(strategy="oneshot", cnn_models=cnn_models)
    for delta in [4, 5]:
        target_stop_states = delta_groups[delta]
        if len(target_stop_states) == 0:
            continue
        print("")
        print(f"Starting to solve delta {delta}: {time.time() - MAIN_START_TIME}")
        predicted_start_states = solver.solve(delta, target_stop_states)
        delta_predicted_start_states[delta] = predicted_start_states
        if EVALUATE:
            print(f"Evaluating delta {delta}: {time.time() - MAIN_START_TIME}")
            stats = Evaluator.evaluate(delta, predicted_start_states, target_stop_states)
            pprint.pprint(stats, indent=4)
            delta_stats[delta] = stats

    print(f"Time since main started: {time.time() - MAIN_START_TIME}")
    print("")

    if EVALUATE:
        print(f"Overall error rate: {np.mean([stats['error_rate'] for stats in delta_stats.values()])}")
        solved_count = 0
        for delta in [1, 2, 3, 4, 5]:
            true_count = len(delta_groups.get(delta, []))
            batched_count = np.ceil(true_count / BATCH_SIZE) * BATCH_SIZE
            solved_count += batched_count
        projected_duration = (
                ((time.time() - MAIN_START_TIME) - load_time)
                * (50000 / solved_count)
                * (5 / len(DELTA_SET)))
        print(f"Projected duration: {projected_duration}")
    print("")

    if SUBMIT:
        Evaluator.submit(delta_indexes, delta_predicted_start_states)