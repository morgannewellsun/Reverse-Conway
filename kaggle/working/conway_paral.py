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

MODEL_ROOT_DIR = '../input/conway/cnn_models/'
CNN_PATHS = {
    "delta_1_final": MODEL_ROOT_DIR + "delta_1_final",
    "delta_1": MODEL_ROOT_DIR + "delta_1",
    # "delta_2": MODEL_ROOT_DIR + "delta_2",
    # "delta_3": MODEL_ROOT_DIR + "delta_3",
    # "delta_4": MODEL_ROOT_DIR + "delta_4",
    # "delta_5": MODEL_ROOT_DIR + "delta_5",
}
KAGGLE_TEST_FILE_PATH = '../input/conways-reverse-game-of-life-2020/test.csv'
OUTPUT_DIR = './'

# Batch size affects both the CNN and the GA.
BATCH_SIZE = 128  # batch size

# The following settings control the CNN.
CNN_USE_DELTA_1_FINAL = True  # whether or not to use an aggressive model for delta = 1

# The following settings control genetic algorithm population initialization.
GA_STATIC_PROB_THRESHOLD_SPAN = 0.5  # half-width of range of thresholds for static
GA_STATIC_POP = 12  # GA initial population from the static prob
GA_DYNAMIC_POP = 50  # GA initial population from the dynamic prob

# The following settings control genetic algorithm dynamics.
GA_ITERATIONS = 100
GA_MUTATION_TYPE = "MUTATION_RATE"
GA_MUTATION_RATE = 0.1
GA_MUTATIONS_PER_BOARD = 5  # unused

# The following settings restricts to only a selected subset of data to test.
DELTA_SET = {1, 2, 3, 4, 5}  # Load only the model for specified deltas. To load all, use {1,2,3,4,5}
GAME_IDX_MIN = 50000  # Kaggle test game indices from 50000 to 99999.
GAME_IDX_MAX = 52000  # To test for 1000 rows, use 51000

# Don't touch.
GA_TOTAL_POP = GA_STATIC_POP + GA_DYNAMIC_POP + 2


# %% [code]
# =====================================================================================================================
# Conway's game logic


class BinaryConwayForwardPropFn:

    def __init__(self, numpy_mode=False, nrows=25, ncols=25):
        self._numpy_mode = numpy_mode
        self.nrows = nrows
        self.ncols = ncols
        self._moore_offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i != 0 or j != 0)]

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

        # print(self._bars_static)                                                                                        # TODO remove

    '''
    def _revert_many(self, model, stop_states):
        # Use CNN to revert many boards by 50% threshold.
        # Arg stop_states is an array of size
        # (number of boards, nrows, ncols, number of guesses).
        cnn_result = model(stop_states.astype(np.float32)).numpy()
        return cnn_result >= 0.5
    '''

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 25, 25, 1), dtype=tf.bool)])
    def _revert_static(self, stop_state) -> tf.Tensor:
        cnn_result = self._model(tf.cast(stop_state, tf.float32))  # (batch, 25, 25, 1)
        x = tf.greater_equal(cnn_result, self._bars_static)  # (pop, batch, 25, 25, 1)
        return x

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 25, 25, 1), dtype=tf.bool)])
    def _revert_dynamic(self, stop_state) -> tf.Tensor:
        cnn_result = self._model(tf.cast(stop_state, tf.float32))  # (batch, 25, 25, 1)
        noise = tf.random.uniform(
            shape=(self._population_size_dynamic, 1, 25, 25, 1),
            minval=0, maxval=1,
            dtype=tf.dtypes.float32)
        x = tf.greater_equal(cnn_result, noise)  # (pop, batch, 25, 25, 1)
        return x

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 25, 25, 1), dtype=tf.bool)])
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
        else:
            raise NotImplementedError

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(GA_TOTAL_POP, None, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, None, 25, 25, 1), dtype=tf.bool)])
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
        else:
            raise NotImplementedError
        muters = tf.random.uniform(
            shape=(GA_TOTAL_POP, 1, 25, 25, 1),
            minval=0, maxval=1,
            dtype=tf.dtypes.float32)  # (pop, 1, 25, 25, 1)
        muters = tf.greater_equal(muter_threshold, muters)  # (pop, batch, 25, 25, 1)
        muters &= target_areas  # (pop, batch, 25, 25, 1)
        return tf.math.logical_xor(population, muters)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(GA_TOTAL_POP, None, 25, 25, 1), dtype=tf.bool)])
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
        tf.TensorSpec(shape=(None, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, None, 25, 25, 1), dtype=tf.bool)])
    def _calculate_diffs_and_match_counts(
            self, target: tf.Tensor, population: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        for _ in range(self._delta):
            population = self._conway(population)
        diffs = tf.math.logical_xor(population, target)  # (pop, batch, 25, 25, 1)
        match_counts = (
                tf.constant(625, dtype=tf.int32)
                - tf.transpose(tf.math.count_nonzero(diffs, axis=(-3, -2, -1), dtype=tf.int32)))  # (batch, pop)
        return diffs, match_counts

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, None, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, None, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(None, GA_TOTAL_POP), dtype=tf.int32)])
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
        tf.TensorSpec(shape=(None, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, None, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, None, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(None, GA_TOTAL_POP), dtype=tf.int32)])
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
        tf.TensorSpec(shape=(None, 25, 25, 1), dtype=tf.bool),
        tf.TensorSpec(shape=(GA_TOTAL_POP, None, 25, 25, 1), dtype=tf.bool)])
    def refine(self, target, population) -> tf.Tensor:
        population_diffs, population_match_counts = self._calculate_diffs_and_match_counts(target, population)
        for _ in range(self._iterations - 1):
            population, population_diffs, population_match_counts = self._iterate(
                target, population, population_diffs, population_match_counts)
        return self._iterate_last(target, population, population_diffs, population_match_counts)

# %% [code]
# =====================================================================================================================
# Solver strategy definitions.


class Solver:

    def __init__(self, strategy: str, cnn_models: Dict[str, tf.function]):
        self._strategy = strategy
        self._cnn_models = cnn_models

    @staticmethod
    def _collect_into_batches(target_stop_states: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
        batches = []
        batch_list = []
        for target_stop_state in target_stop_states:
            batch_list.append(target_stop_state)
            if len(batch_list) == batch_size:
                batches.append(np.stack(batch_list))
                batch_list = []
        if len(batch_list) > 0:
            batches.append(np.stack(batch_list))

        # print("Batches\n", len(batches), "\n", batches[0].shape, "\n", batches[-1].shape)                               # TODO remove

        print(f"Data split into batches: {time.time() - MAIN_START_TIME}")
        return batches

    def solve(self, delta: int, target_stop_states: List[np.ndarray]) -> List[np.ndarray]:
        if self._strategy == "stackwise":
            return self._solve_stackwise(delta, target_stop_states)
        else:
            raise NotImplementedError

    def _solve_stackwise(self, delta: int, target_stop_states: List[np.ndarray]) -> List[np.ndarray]:
        # target_stop_states: [(25, 25, 1)]
        # returns: [(25, 25, 1)]

        batches = Solver._collect_into_batches(target_stop_states, BATCH_SIZE)  # [(batch, 25, 25, 1)]
        batches = [tf.constant(batch, dtype=bool) for batch in batches]

        cnn_reverter = None
        ga_refiner = GA(iterations=GA_ITERATIONS, delta=delta)

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
                # print("CNN\n", out.get_shape())                                                                         # TODO remove
                cnn_output_batches.append(out)
            print(f"CNN used to revert from delta {delta} to {delta - 1}: {time.time() - MAIN_START_TIME}")
            ga_output_batches = []
            for batch, cnn_output_batch in zip(batches, cnn_output_batches):
                out = ga_refiner.refine(batch, cnn_output_batch)
                # print("GA\n", out.get_shape())                                                                          # TODO remove
                ga_output_batches.append(out)
            print(f"GA used to refine CNN predictions for {delta - 1}: {time.time() - MAIN_START_TIME}")
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
        output_batches = []
        for batch, cnn_output_batch in zip(batches, cnn_output_batches):
            output_batches.append(ga_refiner.refine(batch, cnn_output_batch))
        print(f"GA used to refine CNN predictions for predicted start: {time.time() - MAIN_START_TIME}")

        output = []
        for output_batch in output_batches:
            output.extend(list(output_batch.numpy()))
        return output


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


# %% [code]
# =====================================================================================================================
# Main function.

MAIN_START_TIME = None

if __name__ == "__main__":

    MAIN_START_TIME = time.time()
    print("")

    cnn_models = dict([(name, tf.saved_model.load(path)) for name, path in CNN_PATHS.items()])
    print(f"CNN models loaded: {time.time() - MAIN_START_TIME}")

    data = pd.read_csv(KAGGLE_TEST_FILE_PATH, index_col=0, dtype='int')
    print(f"Data loaded: {time.time() - MAIN_START_TIME}")

    delta_groups = {1: [], 2: [], 3: [], 4: [], 5: []}
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
    print(f"Data parsed to numpy: {time.time() - MAIN_START_TIME}")

    solver = Solver(strategy="stackwise", cnn_models=cnn_models)
    for delta, target_stop_states in delta_groups.items():
        print("")
        print(f"Starting to solve delta {delta}: {time.time() - MAIN_START_TIME}")
        predicted_start_states = solver.solve(delta, target_stop_states)
        print(f"Evaluating delta {delta}: {time.time() - MAIN_START_TIME}")
        stats = Evaluator.evaluate(delta, predicted_start_states, target_stop_states)
        pprint.pprint(stats, indent=4)


    """
    delta_groups = {1: [], 2: [], 3: [], 4: [], 5: []}
    delta_stats = {1: [], 2: [], 3: [], 4: [], 5: []}
    for idx, row in data.iterrows():
        if idx < GAME_IDX_MIN:
            continue
        if idx > GAME_IDX_MAX:
            break
        delta = row[0]
        if delta not in DELTA_SET:
            continue
        tf_arr = np.array(row[1:]).astype(np.float32).reshape((25, 25, 1))
        delta_groups[delta].append(tf_arr)
        delta_stats[delta].append((idx, sum(tf_arr)))   # sunheng

    for delta in [1]:
        np_problems = np.stack(delta_groups[delta])  # (n, 25, 25, 1)

        # TODO FIX BATCHING
        batches = np.array_split(np_problems, len(delta_groups[delta]) // BATCH_SIZE, axis=0)  # [(b, 25, 25, 1)]

        for np_batch in batches:
            my_target = np_batch
            my_stop_state = tf.constant(np_batch, dtype=tf.bool)
            output = my_cnn.revert(my_stop_state)
            break

    my_ga = GA(iterations=10, delta=1, mutations_per_board=5)
    ga_output = my_ga.refine(my_target, output)
    print(ga_output)
    """

"""
# %% [code]
# =====================================================================================================================
# Evaluation tools.


def evaluation(stop_state: np.ndarray, start_state: np.ndarray) -> None:
    # stop_state.shape = (None, 25, 25, 1)
    # start_state.shape = (None, 25, 25, 1)
    pass


output_dir = ''


def eval_result(delta_stats, delta_solvers, delta_errors) -> None:
    # delta_stats: dict from delta to list of (game index, target lives)
    # delta_solvers: dict from delta to game solvers of shape (1, games, 25, 25, 1)
    # delta_errors: dict from delta to (solver livers, solver errors)
    result = []
    submissions = []
    for d in (1, 2, 3, 4, 5):
        for a, b, c in zip(delta_stats[d], delta_errors[d], delta_solvers[d]):
            result.append([d, *a, *b])
            subm = [a[0], *(c.flatten().astype(int).tolist())]
            submissions.append(subm)

    game_size = 25 * 25
    cols = ['id']
    cols.extend(['start_' + str(j) for j in range(game_size)])
    pd.DataFrame(submissions, columns=cols).to_csv(output_dir + 'submission.csv', index=False)

    with pd.ExcelWriter(output_dir + 'run_stats.xlsx', engine='xlsxwriter') as writer:
        data = np.DataFrame(result, columns=(
            'delta', 'game index', 'target lives', 'solve lives', 'solve errors'))
        data.to_excel(writer, sheet_name='result')

        # Generate more statistical reports based on the above data.
        # statistics by errors
        err_col = ['delta ' + str(j) for j in range(6)]
        err_stats = pd.DataFrame([[0] * 6] * game_size, columns=err_col)
        # statistics by number of lives at the end state
        liv_stats = pd.DataFrame([[0] * 2] * game_size, columns=('count', 'fails'))
        del_stats = pd.DataFrame([[0] * 3] * 6, columns=('count', 'hits', 'fails'))

        for j, row in data.iterrows():
            (delta, game_index, target_lives,
             solv_lives, solv_errors) = map(int, row)

            err_stats.iloc[solv_errors, delta] += 1
            liv_stats.iloc[target_lives, 0] += 1
            liv_stats.iloc[target_lives, 1] += solv_errors
            del_stats.iloc[delta, 0] += 1
            del_stats.iloc[delta, 2] += solv_errors
            if solv_errors == 0:
                del_stats.iloc[delta, 1] += 1

        err_stats['total'] = err_stats.sum(axis=1)
        err_stats = err_stats.loc[err_stats['total'] > 0, :]
        err_stats.to_excel(writer, sheet_name='errors')

        liv_stats = liv_stats.loc[liv_stats['count'] > 0, :]
        liv_stats['accuracy'] = 1 - liv_stats['fails'] / liv_stats['count'] / game_size
        liv_stats.to_excel(writer, sheet_name='lives')

        del_stats = del_stats[del_stats.index > 0]
        del_stats['accuracy'] = 1 - del_stats['fails'] / del_stats['count'] / game_size
        del_stats.to_excel(writer, sheet_name='deltas')

"""
'''
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
    pd.DataFrame(all_submissions, columns=cols).to_csv(OUTPUT_DIR + 'submission.csv', index=False)

    with pd.ExcelWriter(OUTPUT_DIR + 'run_stats.xlsx', engine='xlsxwriter') as writer:
        # Record basic settings for later review with results.
        pd.DataFrame([
            ['batch_size', BATCH_SIZE],
            ['cnn_paths', CNN_PATHS],
            ['deltaset', DELTA_SET],
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
            ['game_idx_min', GAME_IDX_MIN],
            ['game_idx_max', GAME_IDX_MAX],
            ['rand_seed', RANDOM_SEED],
            ['stepwise', stepwise],
            ['track_details', track_details],
            ['start_time', start_time],
            ['end_time', end_time]
        ], columns=('key', 'value')
        ).to_excel(writer, sheet_name='config')
        data = pd.DataFrame(all_results, columns=result_header)
        data.to_excel(writer, sheet_name='result')

        # Generate more statistical reports based on the above data.
        game_size = 25 * 25
        # statistics by errors
        err_col = ['delta ' + str(j) for j in range(6)]
        err_stats = pd.DataFrame([[0] * 6] * game_size, columns=err_col)
        # statistics by number of lives at the end state
        liv_stats = pd.DataFrame([[0] * 3] * game_size, columns=(
            'count', 'cnn_fails', 'ga_fails'))
        del_stats = pd.DataFrame([[0] * 5] * 6, columns=(
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
        err_stats = err_stats.loc[err_stats['total'] > 0, :]
        err_stats.to_excel(writer, sheet_name='errors')

        liv_stats = liv_stats.loc[liv_stats['count'] > 0, :]
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
# Load CNN models, test file, and set up GA.# Load CNN models, test file, and set up GA.

myprev = {'cnn_static': 0, 'cnn_dynamic': 0, 'cnn_total': 0, 'cnn_many': 0, 'ga_total': 0}
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
cnn_manager = CnnMan(cnn_solver, stepwise)
mylog('CNN models loaded')


#### Load Kaggle test files
data = pd.read_csv(KAGGLE_TEST_FILE_PATH, index_col=0, dtype='int')
mylog('Kaggle file loaded')

#### Apply GA to improve.
np.random.seed(RANDOM_SEED)
conway = BinaryConwayForwardPropFn(numpy_mode=True, nrows=25, ncols=25)
ga = ReverseGa(conway, pop_size=ga_pop_size, max_iters=ga_max_iters,
               crossover_rate=ga_cross, mutation_rate=ga_mutate,
               mut_div=ga_mut_div,
               tracking=track_details, save_states=ga_save_states)

# %% [code]
# Actual run

mylog('GA run started')

all_results = []
all_submissions = []
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

delta_groups = {1: [], 2: [], 3: [], 4: [], 5: []}
for idx, row in data.iterrows():
    if idx < GAME_IDX_MIN:
        continue
    if idx > GAME_IDX_MAX:
        break
    delta = row[0]
    if delta not in DELTA_SET:
        continue
    tf_arr = np.array(row[1:]).astype(np.float32).reshape((25, 25, 1))
    delta_groups[delta].append(tf_arr)

for delta in [1, 2, 3, 4, 5]:
    np_problems = np.stack(delta_groups[delta])  # (n, 25, 25, 1)
    batches = np.array_split(np_problems, len(delta_groups[delta]) // BATCH_SIZE, axis=0)  # [(b, 25, 25, 1)]
    solved_batches = []
    for np_batch in batches:
        solved_batches.append(cnn_manager.revert(
            np_batch, delta, ga_static_1, ga_static_n, ga_dynamic_1, ga_dynamic_n))

"stop_state, delta, static_1, static_n, dynamic_1, dynamic_n)"

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
'''
