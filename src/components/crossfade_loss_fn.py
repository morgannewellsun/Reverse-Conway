import sys
from typing import *

from components.true_target_loss_fn import TrueTargetLossFn

import tensorflow as tf


class CrossfadeLossFn(tf.keras.losses.Loss):

    def __init__(
            self,
            epochs_initial: int,
            epochs_transition: int,
            final_fade_in_weight: float,
            name: Optional[str] = None,
            verbose: bool = False):
        super(CrossfadeLossFn, self).__init__(name=name)
        self._config = {
            "epochs_initial": epochs_initial,
            "epochs_transition": epochs_transition,
            "final_fade_in_weight": final_fade_in_weight,
            "name": name,
            "verbose": verbose}
        self._loss_fn_initial = tf.keras.losses.BinaryCrossentropy()
        self._loss_fn_final = TrueTargetLossFn(delta_steps=1, y_true_is_start=True)
        self._slope = tf.constant(-1 * final_fade_in_weight / epochs_transition, dtype=tf.float32)
        self._intersect = tf.constant(1 - epochs_initial * self._slope, dtype=tf.float32)
        self._lower_bound = 1 - final_fade_in_weight
        self.epochs_seen = tf.Variable(initial_value=0.0, dtype=tf.float32)
        self._verbose = verbose

    def call(self, y_true, y_pred):
        loss_initial = self._loss_fn_initial(y_true, y_pred)
        loss_final = self._loss_fn_final(y_true, y_pred)
        weight_initial = tf.clip_by_value(
            tf.add(tf.scalar_mul(self.epochs_seen, self._slope), self._intersect),
            self._lower_bound, 1.0)
        weight_final = tf.subtract(1.0, weight_initial)
        if self._verbose:
            tf.print("", output_stream=sys.stdout)
            tf.print("Epochs seen: ", self.epochs_seen, output_stream=sys.stdout)
            tf.print("Losses: ", loss_initial, loss_final, output_stream=sys.stdout)
            tf.print("Weights: ", weight_initial, weight_final, output_stream=sys.stdout)
        weighted_average_loss = tf.add(
            tf.scalar_mul(loss_initial, weight_initial),
            tf.scalar_mul(loss_final, weight_final))
        return weighted_average_loss

    def get_config(self):
        return self._config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EpochsSeenUpdaterCallback(tf.keras.callbacks.Callback):

    def __init__(self, output_variable: tf.Variable):
        super(EpochsSeenUpdaterCallback, self).__init__()
        self._output_variable = output_variable

    def on_epoch_begin(self, epoch, logs=None):
        self._output_variable.assign(epoch)





