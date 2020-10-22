from typing import *

import numpy as np
import tensorflow as tf

from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn
from data.pretty_test_target import pretty_test_target
from framework.visualizer import Visualizer


class VisualizationCallback(tf.keras.callbacks.Callback):

    def __init__(
            self,
            *,
            test_batches: tf.keras.utils.Sequence,
            delta_steps: int,
            output_directory: str,
            epochs_per_visualization: int,
            visualize_first_n: int = 1,
            binary_threshold: float = 0.5,
    ):
        super(VisualizationCallback, self).__init__()
        self._test_batches = test_batches
        self._delta_steps = delta_steps
        self._output_directory = output_directory
        self._epochs_per_visualization = epochs_per_visualization
        self._visualize_first_n = visualize_first_n
        self._compare_end = delta_steps != 0
        self._binary_threshold = binary_threshold
        self._epochs_since_last_visualization = np.inf
        self._visualizer = Visualizer(show_figures=False, save_directory=output_directory)
        self._binary_forward_prop = BinaryConwayForwardPropFn(numpy_mode=True)

    def on_epoch_end(self, epoch, logs=None):
        self._epochs_since_last_visualization += 1
        if self._epochs_since_last_visualization >= self._epochs_per_visualization:
            self._epochs_since_last_visualization = 0
            for i in range(self._visualize_first_n):
                end_board_true = self._test_batches[0][0][i:i+1]
                start_board_pred = np.greater_equal(self.model.predict(end_board_true), self._binary_threshold)
                if self._compare_end:
                    end_board_pred = start_board_pred
                    for _ in range(self._delta_steps):
                        end_board_pred = self._binary_forward_prop(end_board_pred)
                    self._visualizer.draw_board_comparison(
                        end_board_true.squeeze(), end_board_pred.squeeze(), title=f"epoch_{epoch}_board_{i}_end")
                else:
                    start_board_true = self._test_batches[0][1][i:i+1]
                    self._visualizer.draw_board_comparison(
                        start_board_true.squeeze(), start_board_pred.squeeze(), title=f"epoch_{epoch}_board_{i}_start")

