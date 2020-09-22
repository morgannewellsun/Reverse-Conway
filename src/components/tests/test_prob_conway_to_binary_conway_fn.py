import unittest

import numpy as np

from components.prob_conway_to_binary_conway_fn import ProbConwayToBinaryConwayFn


class TestProbConwayToBinaryConwayFn(unittest.TestCase):

    def test_call(self):
        for _ in range(100):
            test_input = np.random.random((10, 10))
            fn = ProbConwayToBinaryConwayFn(threshold=0.5)
            test_output = fn(test_input).numpy()
            expected_output = test_input >= 0.5
            self.assertTrue(np.array_equal(expected_output, test_output))


if __name__ == "__main__":
    unittest.main()
