import unittest

import numpy as np

from components.prob_conway_forward_prop_fn import ProbConwayForwardPropFn


class TestProbConwayForwardLayerFn(unittest.TestCase):

    def test_generate_indices_and_complements_8cr(self):
        sum_one_to_eight = sum(range(8))
        for r in range(8):
            indices, complements = ProbConwayForwardPropFn._generate_indices_and_complements_8cr(r)
            print(indices)
            print(complements)
            print("__")
            for i, c in zip(indices, complements):
                self.assertEqual(r, len(i))
                self.assertEqual(8-r, len(c))
                self.assertEqual(sum(i + c), sum_one_to_eight)

    def test_call_unbatched(self):
        layer = ProbConwayForwardPropFn()
        test_input = 0.5 * np.array(
            [[0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]]).astype(np.float32)
        test_input = np.expand_dims(test_input, axis=-1)
        test_output = layer(test_input).numpy()
        a = 0.125
        b = 0.3125
        expected_output = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [a, b, b, 0, 0, 0, 0],
             [0, b, a, 0, 0, 0, 0],
             [0, a, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, b, b, 0],
             [0, 0, 0, 0, b, b, 0],
             [0, 0, 0, 0, 0, 0, 0]])
        expected_output = np.expand_dims(expected_output, axis=-1)
        self.assertTrue(np.array_equal(expected_output, test_output))

    def test_call_batched(self):
        layer = ProbConwayForwardPropFn()
        test_input = 0.5 * np.array(
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
              [0, 0, 0, 0, 0, 0, 0]]]).astype(np.float32)
        test_input = np.expand_dims(test_input, axis=-1)
        test_output = layer(test_input).numpy()
        a = 0.125
        b = 0.3125
        expected_output = np.array(
            [[[0, 0, 0, 0, 0, 0, 0],
              [a, b, b, 0, 0, 0, 0],
              [0, b, a, 0, 0, 0, 0],
              [0, a, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, b, b, 0],
              [0, 0, 0, 0, b, b, 0],
              [0, 0, 0, 0, 0, 0, 0]],
             [[0, 0, 0, 0, 0, 0, 0],
              [a, b, b, 0, 0, 0, 0],
              [0, b, a, 0, 0, 0, 0],
              [0, a, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]]])
        expected_output = np.expand_dims(expected_output, axis=-1)
        self.assertTrue(np.array_equal(expected_output, test_output))


if __name__ == "__main__":
    unittest.main()
