# Test Game of Life library code for various settings.

import random
from gamelib.conwaymap import ConwayMap
from gamelib.logutil import init_game_log


init_game_log('Test game of life')
cm = ConwayMap(100, 100)
start_state = random.randint(1, 2**cm.size-1)
# print(cm.run(start_state))
cm.animate(start_state, 500, rest=0.2)
