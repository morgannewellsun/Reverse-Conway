import sys
from typing import *

import tensorflow as tf


class CrossfadeLossFn(tf.keras.losses.Loss):

    def __init__(
            self,
            loss_fns: List[tf.keras.losses.Loss],
            weight_fns: List[Callable],
            batches_per_epoch: int,
            name: Optional[str] = None,
            verbose: bool = False):
        super(CrossfadeLossFn, self).__init__(name=name)
        assert len(loss_fns) > 1
        assert len(loss_fns) == len(weight_fns)
        self._loss_fns = loss_fns
        self._weight_fns = weight_fns
        self._batches_per_epoch = batches_per_epoch
        self._batches_seen = tf.Variable(initial_value=0)
        self._verbose = verbose

    def call(self, y_true, y_pred):
        losses = [loss_fn(y_true, y_pred) for loss_fn in self._loss_fns]
        epochs_seen = self._batches_seen, self._batches_per_epoch
        weights = [weight_fn(epochs_seen) for weight_fn in self._weight_fns]
        weighted_losses = [tf.math.multiply(losses[i], weights[i]) for i in range(len(losses))]
        weighted_sum_of_losses = tf.math.add_n(weighted_losses)
        if self._verbose:
            tf.print(
                f"Loss weights with {epochs_seen} epochs seen: "
                f"{[weight for weight in weights]}", output_stream=sys.stdout)
        # self._batches_seen = tf.math.add(self._batches_seen, tf.constant(1))
        return weighted_sum_of_losses



