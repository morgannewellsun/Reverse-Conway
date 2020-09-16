# Test Kaggle's data
# Verify start state will arrive stop state in Kaggle's train file.

import csv
import logging
from gamelib.gamelife import GameLife
from gamelib.logutil import init_game_log, kaggle_root


init_game_log('Kaggle data check')
side_len = 25
board_size = side_len * side_len

matched = True
still_cnt = 0
with open(kaggle_root + 'train.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        # Each row has values: index, delta, start * 625, stop * 625.
        game_idx, delta = int(row[0]), int(row[1])
        start_arr = row[2:(board_size+2)]
        s_brd = [int(j) for j in start_arr]
        stop_arr = row[(board_size+2):(2*board_size+2)]
        k_brd = [int(j) for j in stop_arr]
        
        gl = GameLife(side_len, side_len)
        still_cnt += gl.run(start_array = s_brd, iterations = int(delta))
        m_brd = gl.last_array()
        if not k_brd == m_brd:
            matched = False
            print('Failed to match game {}:'.format(game_idx))
            print('K: {}'.format(k_brd))
            print('M: {}'.format(m_brd))
            break
        if game_idx % 1000 == 0:
            print('Matched {} games with {} still.'.format(game_idx+1, still_cnt))

if matched:
    msg = 'All {} records are matched with {} still.'.format(game_idx+1, still_cnt)
    print(msg)
    logging.info(msg)
