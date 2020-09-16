# Visual inspection of reverse game solver results.

import pandas as pd
import matplotlib.pyplot as plt
from gamelib.gamelife import GameLife
from gamelib.logutil import kaggle_root, output_root

max_csv_rows = 100

gl = GameLife(25, 25)

stop_states = pd.read_csv(kaggle_root + 'test.csv', index_col=0, dtype='int', nrows=max_csv_rows)
start_states = pd.read_csv(output_root + 'nn_predict/predict.csv', nrows=max_csv_rows)

plt.ion()
for row in range(max_csv_rows):
    gl.visual_error(start_states.iloc[row, 2:], stop_states.iloc[row,0], stop_states.iloc[row, 1:])
    plt.pause(1)
    
