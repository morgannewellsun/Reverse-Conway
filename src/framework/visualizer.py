import os.path
from typing import *

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:

    def __init__(
            self,
            show_figures: bool, save_directory: Optional[str],
            figure_size: Tuple[float, float] = (4, 4),
            color_dead: str = "white",
            color_live: str = "black",
            color_true_dead: str = "lightgrey",
            color_true_live: str = "lightgreen",
            color_false_dead: str = "blue",
            color_false_live: str = "red",
            binary_threshold: float = 0.5
    ):
        self._show_figures = show_figures
        self._save_directory = save_directory
        self._figure_size = figure_size
        self._binary_threshold = float(binary_threshold)
        self._cmap_binary = matplotlib.colors.ListedColormap([color_dead, color_live])
        self._norm_binary = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5], self._cmap_binary.N, clip=True)
        self._cmap_comparison = matplotlib.colors.ListedColormap(
            [color_true_dead, color_false_dead, color_false_live, color_true_live])
        self._norm_comparison = matplotlib.colors.BoundaryNorm(
            [-0.5, 0.5, 1.5, 2.5, 3.5], self._cmap_comparison.N, clip=True)


    def draw_board(self, board: np.ndarray, title: str):
        fig, ax = plt.subplots(figsize=self._figure_size)
        board = board.astype(float)
        board = 3.0 * np.greater_equal(board, self._binary_threshold)
        ax.imshow(board, interpolation="nearest", cmap=self._cmap_binary, norm=self._norm_binary)
        self._finalize_and_output_current_fig(title)


    def draw_board_comparison(self, board_true: np.ndarray, board_pred: np.ndarray, title: str):
        fig, ax = plt.subplots(figsize=self._figure_size)
        board_true = board_true.astype(float)
        board_pred = board_pred.astype(float)
        board_true = np.greater_equal(board_true, self._binary_threshold)
        board_pred = np.greater_equal(board_pred, self._binary_threshold)
        board_combined = (board_pred * 2.0) + board_true
        ax.imshow(board_combined, interpolation="nearest", cmap=self._cmap_comparison, norm=self._norm_comparison)
        self._finalize_and_output_current_fig(title)


    def _finalize_and_output_current_fig(self, title: str):
        plt.title(title)
        plt.tight_layout()
        if self._save_directory is not None:
            file_name = title + ".png"
            plt.savefig(os.path.join(self._save_directory, file_name))
        if self._show_figures:
            plt.show()
        plt.close()

    
    # A convenicne method taking board of, e.g., (1, 25, 25, 1) array.
    def draw_board_arr(self, board, title):
        mid2d = (board.shape[1], board.shape[2])
        brd = board.reshape(mid2d)
        self.draw_board(brd, title)


    # A convenicne method taking board of, e.g., (1, 25, 25, 1) array.
    def draw_board_comparison_arr(self, board_true, board_pred, title):
        mid2d = (board_true.shape[1], board_true.shape[2])
        self.draw_board_comparison(
            board_true.reshape(mid2d), board_pred.reshape(mid2d), title)



if __name__ == "__main__":
    from data.pretty_test_target import pretty_test_target
    test_board = pretty_test_target.squeeze()
    visualizer = Visualizer(show_figures=True, save_directory=None)
    visualizer.draw_board(test_board, "test_board")
