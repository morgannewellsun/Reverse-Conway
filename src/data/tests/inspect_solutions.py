import pandas as pd
import numpy as np
from framework.visualizer import Visualizer
from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn


# Run src/single.py. Generate result file results.csv in the dir below.
output_dir = '../../gamelife_data/output/'
data = pd.read_csv(output_dir + 'results.csv', index_col=0)

game_lookfor = 50007
for j, row in data.iterrows():
    if j == game_lookfor:
        break

(game_index, delta, target_lives, cnn_lives, cnn_errors,
 ga_lives, ga_errors) = map(int, row[:7])
(target, cnn, ga) = row[7:]

end_state = np.array(list(target)).astype(bool).reshape((1,25,25,1))
ga_state = np.array(list(ga)).astype(bool).reshape((1,25,25,1))
if not cnn == 0:
    cnn_state = np.array(list(cnn)).astype(bool).reshape((1,25,25,1))

print(game_index)

visual = Visualizer(show_figures=True, save_directory=None)
visual.draw_board_arr(ga_state, 'GA Start')
visual.draw_board_arr(end_state, 'End State')

binary_prop = BinaryConwayForwardPropFn(numpy_mode=True)
ga_end = binary_prop(ga_state, delta)
visual.draw_board_arr(ga_end, 'GA End')

visual.draw_board_comparison_arr(end_state, ga_end, 'GA Errors')
