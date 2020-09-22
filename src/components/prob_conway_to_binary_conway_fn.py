import itertools
import tensorflow as tf


class ProbConwayToBinaryConwayFn:

    def __init__(self, threshold: float):
        self._threshold = threshold

    def __call__(self, inputs):
        return tf.math.greater_equal(inputs, self._threshold)

