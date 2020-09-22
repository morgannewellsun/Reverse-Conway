import unittest

import numpy as np

from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn


class TestBinaryConwayForwardLayerFn(unittest.TestCase):

    def test_call_unbatched(self):
        layer = BinaryConwayForwardPropFn()
        test_input = np.array(
            [[0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]]).astype(np.bool)
        test_input = np.expand_dims(test_input, axis=-1)
        test_output = layer(test_input).numpy()
        a = 1
        expected_output = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [a, 0, a, 0, 0, 0, 0],
             [0, a, a, 0, 0, 0, 0],
             [0, a, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, a, a, 0],
             [0, 0, 0, 0, a, a, 0],
             [0, 0, 0, 0, 0, 0, 0]])
        expected_output = np.expand_dims(expected_output, axis=-1)
        self.assertTrue(np.array_equal(expected_output, test_output))

    def test_call_batched(self):
        layer = BinaryConwayForwardPropFn()
        test_input = np.array(
            [[[0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0],
              [0, 0, 0, 0, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0]],
             [[0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]]]).astype(np.bool)
        test_input = np.expand_dims(test_input, axis=-1)
        test_output = layer(test_input).numpy()
        a = 1
        expected_output = np.array(
            [[[0, 0, 0, 0, 0, 0, 0],
              [a, 0, a, 0, 0, 0, 0],
              [0, a, a, 0, 0, 0, 0],
              [0, a, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, a, a, 0],
              [0, 0, 0, 0, a, a, 0],
              [0, 0, 0, 0, 0, 0, 0]],
             [[0, 0, 0, 0, 0, 0, 0],
              [a, 0, a, 0, 0, 0, 0],
              [0, a, a, 0, 0, 0, 0],
              [0, a, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]]])
        expected_output = np.expand_dims(expected_output, axis=-1)
        self.assertTrue(np.array_equal(expected_output, test_output))

    def test_numpy_mode(self):
        layer = BinaryConwayForwardPropFn(numpy_mode=True)
        test_input = np.array(
            [[[0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0],
              [0, 0, 0, 0, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0]],
             [[0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]]]).astype(np.bool)
        test_input = np.expand_dims(test_input, axis=-1)
        test_output = layer(test_input)
        a = 1
        expected_output = np.array(
            [[[0, 0, 0, 0, 0, 0, 0],
              [a, 0, a, 0, 0, 0, 0],
              [0, a, a, 0, 0, 0, 0],
              [0, a, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, a, a, 0],
              [0, 0, 0, 0, a, a, 0],
              [0, 0, 0, 0, 0, 0, 0]],
             [[0, 0, 0, 0, 0, 0, 0],
              [a, 0, a, 0, 0, 0, 0],
              [0, a, a, 0, 0, 0, 0],
              [0, a, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]]])
        expected_output = np.expand_dims(expected_output, axis=-1)
        self.assertTrue(np.array_equal(expected_output, test_output))


if __name__ == "__main__":
    unittest.main()
