from typing import *

import numpy as np
import tensorflow as tf


"""
WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP 
WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP 
WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP 
WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP 
WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP 
WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP 
WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP 
"""


class KaggleSupervisedDeltaOneDataGenerator(tf.keras.utils.Sequence):

    def __init__(
            self,
            batch_size: int,
            samples_per_epoch: int,
            standardize_inputs: bool = False,
            density_bounds: Tuple[float, float] = (0.01, 0.99),
            board_size: Tuple[int, int] = (25, 25),
            verbose: bool = False):
        self._batch_size = batch_size
        self._batches_per_epoch = int(samples_per_epoch / batch_size)
        self._standardize_inputs = standardize_inputs
        self._density_bounds = density_bounds
        self._board_size = board_size
        self._verbose = verbose
        self._batches_start = None
        self._batches_stop = None
        self.on_epoch_end()

    def on_epoch_end(self):
        if self._verbose:
            print("Generating a new epoch of randomized data for SUPERVISED training.")
        pass

    def __len__(self):
        self.on_epoch_end()  # https://github.com/tensorflow/tensorflow/issues/35911
        return self._batches_per_epoch

    def __getitem__(self, index):
        if self._standardize_inputs:
            return (self._batches_stop[index] - self._input_mean) / self._input_var, self._batches_start[index]
        else:
            return self._batches_stop[index], self._batches_start[index]
