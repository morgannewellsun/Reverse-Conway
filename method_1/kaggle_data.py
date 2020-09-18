# Test Kaggle's data
# Verify start state will arrive stop state in Kaggle's train file.

import csv
import logging
from gamelib.gamelife import GameLife
from gamelib.logutil import init_game_log, kaggle_root, output_root


init_game_log('Kaggle data check')
side_len = 25
board_size = side_len * side_len

# Count statistics: key is tuple (delta, start_count, end_count)
# value is the number of occurences.
count_stats = dict()
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
        key = (delta, sum(s_brd), sum(k_brd))
        count_stats[key] = count_stats.setdefault(key, 0) + 1
        
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

with open(output_root + 'nn_predict/count_stats.csv', 'w+') as fh:
    fh.write('delta,start,stop,count\n')
    for k, v in count_stats.items():
        fh.write('{},{},{},{}\n'.format(k[0], k[1], k[2], v))
        
