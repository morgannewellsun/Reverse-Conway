from typing import *

import numpy as np
import tensorflow as tf


class BaselineDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, batch_size: int, board_size: Tuple[int, int] = (25, 25), batches_per_epoch: int = 1000):
        self._board_size = board_size
        self._batch_size = batch_size
        self._batches_per_epoch = batches_per_epoch
        self._batches = None
        self.on_epoch_end()

    def on_epoch_end(self):
        self._batches = np.random.randint(2, size=(
            self._batches_per_epoch, self._batch_size, *self._board_size, 1))
        self._batches = self._batches.astype(np.float32)

    def __len__(self):
        return self._batches_per_epoch

    def __getitem__(self, index):
        return self._batches[index], self._batches[index]
