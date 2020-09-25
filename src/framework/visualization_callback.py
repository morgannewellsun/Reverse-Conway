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
            test_batches: Optional[tf.keras.utils.Sequence],
            delta_steps: int,
            output_directory: str,
            epochs_per_visualization: int,
            binary_threshold: float = 0.5,
            use_pretty_test_target: bool = False
    ):
        super(VisualizationCallback, self).__init__()
        if test_batches is None and not use_pretty_test_target:
            raise ValueError("test_batches must be provided if use_pretty_test_target is False.")
        self._test_batches = test_batches
        self._delta_steps = delta_steps
        self._output_directory = output_directory
        self._epochs_per_visualization = epochs_per_visualization
        self._binary_threshold = binary_threshold
        self._use_pretty_test_target = use_pretty_test_target
        self._epochs_since_last_visualization = np.inf
        self._visualizer = Visualizer(show_figures=False, save_directory=output_directory)
        self._binary_forward_prop = BinaryConwayForwardPropFn(numpy_mode=True)

    def on_epoch_end(self, epoch, logs=None):
        self._epochs_since_last_visualization += 1
        if self._epochs_since_last_visualization >= self._epochs_per_visualization:
            self._epochs_since_last_visualization = 0
            end_board_true = pretty_test_target if self._use_pretty_test_target else self._test_batches[0][0][0:1]
            start_board_pred = np.greater_equal(self.model.predict(end_board_true), self._binary_threshold)
            end_board_pred = start_board_pred
            for _ in range(self._delta_steps):
                end_board_pred = self._binary_forward_prop(end_board_pred)
            title = f"epoch_{epoch}_pretty_test_target" if self._use_pretty_test_target else f"epoch_{epoch}_board_0"
            self._visualizer.draw_board_comparison(end_board_true.squeeze(), end_board_pred.squeeze(), title=title)

