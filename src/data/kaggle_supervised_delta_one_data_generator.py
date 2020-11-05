from typing import *

import numpy as np
import tensorflow as tf

from data.kaggle_supervised_data_generator import KaggleSupervisedDataGenerator


class KaggleSupervisedDeltaOneDataGenerator(tf.keras.utils.Sequence):

    def __init__(
            self,
            batch_size: int,
            samples_per_epoch: int,
            warmup_steps_values: Tuple[int, ...] = (5, 6, 7, 8, 9),
            density_bounds: Tuple[float, float] = (0.01, 0.99),
            board_size: Tuple[int, int] = (25, 25),
            attach_epoch_to_y: bool = False,
            verbose: bool = False):
        self._batch_size = batch_size
        self._batches_per_epoch = (
                len(warmup_steps_values) * int(samples_per_epoch / (batch_size * len(warmup_steps_values))))
        self._warmup_steps_values = warmup_steps_values
        self._density_bounds = density_bounds
        self._board_size = board_size
        self._attach_epoch_to_y = attach_epoch_to_y
        self._verbose = verbose
        self._batches_start = None
        self._batches_stop = None
        self._sub_generator_samples_per_epoch = int(batch_size * self._batches_per_epoch / len(warmup_steps_values))
        self._sub_generators = [
            KaggleSupervisedDataGenerator(
                delta_steps=1,
                batch_size=batch_size,
                samples_per_epoch=self._sub_generator_samples_per_epoch,
                standardize_inputs=False,
                warmup_steps=warmup_steps,
                density_bounds=density_bounds,
                board_size=board_size,
                verbose=verbose)
            for warmup_steps in warmup_steps_values]
        self._epochs_generated = 0
        self.on_epoch_end()

    def on_epoch_end(self):
        self._epochs_generated += 1
        if self._verbose:
            print(f"Generating a new epoch of combined delta 1 supervised data "
                  f"({self._batches_per_epoch} batches of size {self._batch_size}).")
        self._batches_start = np.empty(
            (len(self._warmup_steps_values) * self._sub_generator_samples_per_epoch, *self._board_size, 1),
            dtype=np.float32)
        self._batches_stop = np.empty(
            (len(self._warmup_steps_values) * self._sub_generator_samples_per_epoch, *self._board_size, 1),
            dtype=np.float32)
        for i, sub_generator in enumerate(self._sub_generators):
            sub_generator.on_epoch_end()
            sub_batches_start, sub_batches_stop = sub_generator.get_all_batches()
            sub_batches_start = sub_batches_start.reshape(
                (self._sub_generator_samples_per_epoch, *self._board_size, 1))
            sub_batches_stop = sub_batches_stop.reshape(
                (self._sub_generator_samples_per_epoch, *self._board_size, 1))
            self._batches_start[i::len(self._warmup_steps_values)] = sub_batches_start
            self._batches_stop[i::len(self._warmup_steps_values)] = sub_batches_stop
        self._batches_start = self._batches_start.reshape(
            (self._batches_per_epoch, self._batch_size, *self._board_size, 1))
        self._batches_stop = self._batches_stop.reshape(
            (self._batches_per_epoch, self._batch_size, *self._board_size, 1))

    def __len__(self):
        # self.on_epoch_end()  # https://github.com/tensorflow/tensorflow/issues/35911  applies to tf<=2.1
        return self._batches_per_epoch

    def __getitem__(self, index):
        if not self._attach_epoch_to_y:
            return self._batches_stop[index], self._batches_start[index]
        else:
            return self._batches_stop[index], (self._batches_start[index], self._epochs_generated)


if __name__ == "__main__":
    gen = KaggleSupervisedDeltaOneDataGenerator(
        batch_size=10,
        samples_per_epoch=2**10,
        warmup_steps_values=(2, 3, 4),
        verbose=True)
    for stop, start in gen:
        print(stop.shape)
        print(stop.squeeze())
    print(len(gen))
