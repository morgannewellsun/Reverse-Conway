import pandas as pd
import numpy as np
from framework.visualizer import Visualizer
from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn


# Run src/single.py. Generate result file results.csv in the dir below.
output_dir = '../ref/outputs/20201013_complete_ga/'
data = pd.read_csv(output_dir + 'results.csv', index_col=0)

# This is the game number in the first column in the csv.
# It is the Kaggle game index minus 50000.
game_lookfor = 21901
for j, row in data.iterrows():
    if j == game_lookfor:
        break

(game_index, delta, target_lives, cnn_lives, cnn_errors,
 ga_lives, ga_errors) = map(int, row[:7])
(target, cnn, ga) = row[7:]
end_state = np.array(list(target)).astype(bool).reshape((1,25,25,1))
ga_state = np.array(list(ga)).astype(bool).reshape((1,25,25,1))

visual = Visualizer(show_figures=True, save_directory=None)
# The game number in the game board uses Kaggle game index.
if not cnn == 0:
    cnn_state = np.array(list(cnn)).astype(bool).reshape((1,25,25,1))
    visual.draw_board_arr(cnn_state, 'CNN Start - game {} delta {}'.format(game_index, delta))
visual.draw_board_arr(ga_state, 'GA Start - game {} delta {}'.format(game_index, delta))
visual.draw_board_arr(end_state, 'Target End - game {} delta {}'.format(game_index, delta))

binary_prop = BinaryConwayForwardPropFn(numpy_mode=True)
ga_end = binary_prop(ga_state, delta)
visual.draw_board_arr(ga_end, 'GA End - game {} delta {}'.format(game_index, delta))
visual.draw_board_comparison_arr(end_state, ga_end, 'GA Errors - game {} delta {}'.format(game_index, delta))
