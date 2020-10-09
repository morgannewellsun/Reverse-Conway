import unittest

import numpy as np

from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn
from data.kaggle_supervised_data_generator import KaggleSupervisedDataGenerator


class TestBinaryConwayForwardLayerFn(unittest.TestCase):

    def test_correctness(self):
        binary_prop = BinaryConwayForwardPropFn(numpy_mode=True)
        for delta_steps in [1, 2, 3, 4, 5]:
            gen = KaggleSupervisedDataGenerator(delta_steps=delta_steps, batch_size=100, samples_per_epoch=1000)
            for batch_x, batch_y in gen:
                for i in range(len(batch_x)):
                    example_x = batch_x[i]
                    example_y = batch_y[i]
                    for _ in range(delta_steps):
                        example_y = binary_prop(example_y)
                    self.assertTrue(np.array_equal(example_x, example_y))


if __name__ == "__main__":
    unittest.main()
