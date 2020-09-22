import unittest

import numpy as np

from components.roll_padding_2d_layer import RollPadding2DLayer


class TestProbConwayForwardLayer(unittest.TestCase):

    def test_call_unbatched(self):
        test_input = np.array(range(25)).reshape((5, 5, 1))
        layer = RollPadding2DLayer(3)
        test_output = layer(test_input).numpy()
        expected_output = np.array(
            [[12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12],
             [17, 18, 19, 15, 16, 17, 18, 19, 15, 16, 17],
             [22, 23, 24, 20, 21, 22, 23, 24, 20, 21, 22],
             [ 2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2],
             [ 7,  8,  9,  5,  6,  7,  8,  9,  5,  6,  7],
             [12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12],
             [17, 18, 19, 15, 16, 17, 18, 19, 15, 16, 17],
             [22, 23, 24, 20, 21, 22, 23, 24, 20, 21, 22],
             [ 2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2],
             [ 7,  8,  9,  5,  6,  7,  8,  9,  5,  6,  7],
             [12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12]]).reshape((11, 11, 1))
        self.assertTrue(np.array_equal(expected_output, test_output))

    def test_call_batched(self):
        test_input = np.array(range(25)).reshape((1, 5, 5, 1))
        layer = RollPadding2DLayer(3)
        test_output = layer(test_input).numpy()
        expected_output = np.array(
            [[12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12],
             [17, 18, 19, 15, 16, 17, 18, 19, 15, 16, 17],
             [22, 23, 24, 20, 21, 22, 23, 24, 20, 21, 22],
             [ 2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2],
             [ 7,  8,  9,  5,  6,  7,  8,  9,  5,  6,  7],
             [12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12],
             [17, 18, 19, 15, 16, 17, 18, 19, 15, 16, 17],
             [22, 23, 24, 20, 21, 22, 23, 24, 20, 21, 22],
             [ 2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2],
             [ 7,  8,  9,  5,  6,  7,  8,  9,  5,  6,  7],
             [12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12]]).reshape((1, 11, 11, 1))
        self.assertTrue(np.array_equal(expected_output, test_output))


if __name__ == "__main__":
    unittest.main()

