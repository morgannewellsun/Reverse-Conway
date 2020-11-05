import sys
from typing import *

import tensorflow as tf


class CrossfadeLossFn(tf.keras.losses.Loss):

    def __init__(
            self,
            loss_fn_initial: tf.keras.losses.Loss,
            loss_fn_final: tf.keras.losses.Loss,
            epochs_initial: int,
            epochs_transition: int,
            name: Optional[str] = None,
            verbose: bool = False):
        super(CrossfadeLossFn, self).__init__(name=name)
        self._loss_fn_initial = loss_fn_initial
        self._loss_fn_final = loss_fn_final
        self._slope = tf.constant(-1 / epochs_transition, dtype=tf.float32)
        self._intersect = tf.constant((epochs_initial + epochs_transition) / epochs_transition, dtype=tf.float32)
        self.epochs_seen = tf.Variable(initial_value=0.0, dtype=tf.float32)
        self._verbose = verbose

    def call(self, y_true, y_pred):
        loss_initial = self._loss_fn_initial(y_true, y_pred)
        loss_final = self._loss_fn_final(y_true, y_pred)
        weight_initial = tf.clip_by_value(
            tf.add(tf.scalar_mul(self.epochs_seen, self._slope), self._intersect),
            0.0, 1.0)
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


class EpochsSeenUpdaterCallback(tf.keras.callbacks.Callback):

    def __init__(self, output_variable: tf.Variable):
        super(EpochsSeenUpdaterCallback, self).__init__()
        self._output_variable = output_variable

    def on_epoch_begin(self, epoch, logs=None):
        self._output_variable.assign(epoch)





