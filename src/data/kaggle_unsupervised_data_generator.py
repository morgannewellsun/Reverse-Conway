from typing import *

import numpy as np
import tensorflow as tf

from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn


class KaggleUnsupervisedDataGenerator(tf.keras.utils.Sequence):

    delta_means = {
        1: 0.1526,
        2: 0.1494,
        3: 0.1460,
        4: 0.1433,
        5: 0.1404}
    delta_vars = dict()
    for delta, mean in delta_means.items():
        delta_vars.update({delta: (((1-mean) * (mean**2)) + (mean * ((1-mean)**2)))})

    def __init__(
            self,
            delta_steps: int,
            batch_size: int,
            samples_per_epoch: int,
            standardize_inputs: bool = False,
            density_bounds: Tuple[float, float] = (0.01, 0.99),
            board_size: Tuple[int, int] = (25, 25),
            verbose: bool = False):
        self._batch_size = batch_size
        self._batches_per_epoch = int(samples_per_epoch / batch_size)
        self._forward_steps = 5 + delta_steps
        self._standardize_inputs = standardize_inputs
        self._density_bounds = density_bounds
        self._board_size = board_size
        self._verbose = verbose
        self._batches = None
        self._rng = np.random.default_rng()
        self._binary_forward_prop = BinaryConwayForwardPropFn(numpy_mode=True)
        self._input_mean = KaggleUnsupervisedDataGenerator.delta_means[delta_steps]
        self._input_var = KaggleUnsupervisedDataGenerator.delta_vars[delta_steps]
        self.on_epoch_end()

    def on_epoch_end(self):
        if self._verbose:
            print("Generating a new epoch of randomized data for UNSUPERVISED training.")
        self._batches = self._rng.uniform(0, 1, size=(
            self._batches_per_epoch, self._batch_size, *self._board_size, 1))
        thresholds = self._rng.uniform(*self._density_bounds, size=(
            self._batches_per_epoch, self._batch_size, 1, 1, 1))
        self._batches = np.greater_equal(self._batches, thresholds)
        for _ in range(self._forward_steps):
            self._batches = self._binary_forward_prop(self._batches)
        invalid_target = np.logical_not(np.any(self._batches, axis=(-3, -2, -1)))
        while np.any(invalid_target):
            invalid_batch_and_board_indices = list(zip(*np.where(invalid_target)))
            new_boards = self._rng.uniform(0, 1, size=(
                len(invalid_batch_and_board_indices), *self._board_size, 1))
            new_thresholds = self._rng.uniform(*self._density_bounds, size=(
                len(invalid_batch_and_board_indices), 1, 1, 1))
            new_boards = np.greater_equal(new_boards, new_thresholds)
            for _ in range(self._forward_steps):
                new_boards = self._binary_forward_prop(new_boards)
            for i, (batch_ind, board_ind) in enumerate(invalid_batch_and_board_indices):
                self._batches[batch_ind, board_ind] = new_boards[i]
            invalid_target = np.logical_not(np.any(self._batches, axis=(-3, -2, -1)))
        self._batches = self._batches.astype(np.float32)

    def __len__(self):
        self.on_epoch_end()  # https://github.com/tensorflow/tensorflow/issues/35911
        return self._batches_per_epoch

    def __getitem__(self, index):
        if self._standardize_inputs:
            return (self._batches[index] - self._input_mean) / self._input_var, self._batches[index]
        else:
            return self._batches[index], self._batches[index]
