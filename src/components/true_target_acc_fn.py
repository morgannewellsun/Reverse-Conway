from typing import *

import tensorflow as tf

from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn
from components.prob_conway_to_binary_conway_fn import ProbConwayToBinaryConwayFn


class TrueTargetAccFn(tf.keras.losses.Loss):

    def __init__(self, delta_steps: int, name: Optional[str] = None, y_true_is_start: bool = False):
        super(TrueTargetAccFn, self).__init__(name=name)
        self._delta_steps = delta_steps
        self._y_true_is_start = y_true_is_start
        self._prob_to_binary = ProbConwayToBinaryConwayFn(threshold=0.5)
        self._binary_forward_prop = BinaryConwayForwardPropFn()

    def call(self, y_true, y_pred):
        y_pred = self._prob_to_binary(y_pred)
        y_true = tf.cast(y_true, dtype=tf.bool)
        for _ in range(self._delta_steps):
            y_pred = self._binary_forward_prop(y_pred)
            if self._y_true_is_start:
                y_true = self._binary_forward_prop(y_true)
        n_correct = tf.math.count_nonzero(tf.math.equal(y_true, y_pred), dtype=tf.int32)
        n_total = tf.size(y_true)
        true_target_acc = tf.subtract(
            tf.constant(1.0, dtype=tf.float32),
            tf.cast(tf.math.divide(n_correct, n_total), dtype=tf.float32))
        return true_target_acc

    def get_config(self):
        return {"delta_steps": self._delta_steps, "name": self.name}
