from typing import *

import numpy as np
import tensorflow as tf

from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn


class KaggleDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, batch_size: int, batches_per_epoch: int, delta_steps: int):
        self._batch_size = batch_size
        self._batches_per_epoch = batches_per_epoch
        self._forward_steps = 5 + delta_steps
        self._density_bounds = (0.01, 0.99)
        self._board_size = (25, 25)
        self._batches = None
        self._rng = np.random.default_rng()
        self._binary_forward_prop = BinaryConwayForwardPropFn(numpy_mode=True)
        self.on_epoch_end()

    def on_epoch_end(self):
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
        return self._batches_per_epoch

    def __getitem__(self, index):
        return self._batches[index], self._batches[index]
