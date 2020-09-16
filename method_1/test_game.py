# Test Game of Life library code for various settings.

import random
from gamelib.gamelife import GameLife
from gamelib.logutil import init_game_log


def play_state(board_size, init_state, iterations, rest=0.1):
    gl = GameLife(board_size[0], board_size[1])
    gl.run(start_state = init_state, iterations = iterations)
    gl.animate(rest)


def play_board(board_size, board_arr, iterations):
    gl = GameLife(board_size[0], board_size[1])
    gl.run(start_array = board_arr, iterations = iterations)
    gl.animate()


init_game_log('Test game of life')

# In Spider command line, run
# %matplotlib qt5
# Or change IPython Graphics preferences.
if False:
    # Test a slider
    slider = {1:{1}, 2:{2,3}, 3:{1,2}}
    play_state((13, 10), slider, 200)
elif False:
    # Test still/semi still configurations.
    stable = {1:{2,3}, 2:{2,3, 7,8,9}}
    play_state((5, 12), stable, 100)
elif False:
    # Test an expansion.
    expand = {4:{5}, 5:{4,5,6}, 6:{5}}
    play_state((20, 20), expand, 20, 1)
elif False:
    # Some messy figure.
    messy = {1:{2,3}, 2:{2,3, 6,7,8}, 6:{5}, 7:{4,5,6}, 8:{5}}
    play_state((10, 15), messy, 200, 0.5)
elif True:
    # Random figure.
    board_size = (50, 40)
    print('Playing game of life on 100X100 board. It takes a while to form.')
    rand_bd = [random.randint(0, 1) for j in range(board_size[0] * board_size[1])]
    play_board(board_size, rand_bd, 200)

