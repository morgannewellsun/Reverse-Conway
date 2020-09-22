import unittest

import numpy as np
import tensorflow as tf

from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn
from components.true_target_loss_fn import TrueTargetLossFn


class TestTrueTargetLossFn(unittest.TestCase):

    def test_call(self):
        loss = TrueTargetLossFn(delta_steps=2)
        forward = BinaryConwayForwardPropFn()
        for _ in range(100):
            test_start_prob = np.random.random((1, 10, 10, 1))
            test_start_binary = test_start_prob > 0.5
            test_stop_prob = tf.cast(forward(forward(test_start_binary)), dtype=tf.float32)
            print(loss(test_stop_prob, test_start_prob).numpy())
            print(loss(1-test_stop_prob, test_start_prob).numpy())


if __name__ == "__main__":
    unittest.main()
