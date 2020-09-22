import tensorflow as tf

from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn
from components.prob_conway_to_binary_conway_fn import ProbConwayToBinaryConwayFn


class TrueTargetAccFn:

    __name__ = "TrueTargetAcc"

    def __init__(self, delta_steps: int):
        self._delta_steps = delta_steps
        self._prob_to_binary = ProbConwayToBinaryConwayFn(threshold=0.5)
        self._binary_forward_prop = BinaryConwayForwardPropFn()

    def __call__(self, y_true, y_pred):
        y_pred = self._prob_to_binary(y_pred)
        for _ in range(self._delta_steps):
            y_pred = self._binary_forward_prop(y_pred)
        y_true = tf.cast(y_true, dtype=tf.bool)
        n_correct = tf.math.count_nonzero(tf.math.equal(y_true, y_pred), dtype=tf.int32)
        n_total = tf.size(y_true)
        true_target_acc = tf.subtract(
            tf.constant(1.0, dtype=tf.float32),
            tf.cast(tf.math.divide(n_correct, n_total), dtype=tf.float32))
        return true_target_acc
