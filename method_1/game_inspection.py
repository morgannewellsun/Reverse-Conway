# Visual inspection of reverse game solver results.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gamelib.gamelife import GameLife
from gamelib.reversenet import ReverseNet
from gamelib.logutil import kaggle_root, output_root

# %matplotlib qt5

max_csv_rows = 300
start_csv_row = 250

gl = GameLife(25, 25)
stop_states = pd.read_csv(kaggle_root + 'test.csv', index_col=0, dtype='int', nrows=max_csv_rows)
start_states = pd.read_csv(output_root + 'nn_predict/predict.csv', nrows=max_csv_rows)
plt.ion()
for row in range(start_csv_row, max_csv_rows):
    gl.visual_error(start_states.iloc[row, 2:], stop_states.iloc[row,0], stop_states.iloc[row, 1:])
    plt.pause(1)



# # For the train set, compare predicted probabilities against actual value.
# board = (25, 25)
# board_size = np.product(board)
# rn = ReverseNet(board_size, [256, 64, 32, 32, 32, 64], batches = 2048, epochs=100)
# rn.load()
# data = pd.read_csv(kaggle_root + 'train.csv', index_col=0, dtype='int', nrows=max_csv_rows)
# stop_states = data.iloc[start_csv_row:, :(board_size+1)]    # contains delta column
# start_states = data.iloc[start_csv_row:, (board_size+1):]  # does not contain delta
# pred_states = rn.revert(stop_states, tofile = False)[0]   # does not contain delta column

# fig, ax = plt.subplots()
# plt.ion()

# j = 10
# actual = np.array(start_states.iloc[j, :]).reshape((25, 25))
# predicted = pred_states[j].reshape((25, 25))
# im = ax.imshow(actual, cmap = plt.cm.plasma)
# for c in range(25):
#     for r in range(25):
#         ax.text(r, c, round(predicted[r][c]*100), ha="center", va="center")
# t = 'Predicted vs Actual for {} delta {}'.format(start_states.index[j], stop_states.iloc[j,0])
# ax.set_title(t)
# fig.tight_layout()
# plt.show()
# plt.pause(600)
