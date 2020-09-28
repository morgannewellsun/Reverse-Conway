# Test Kaggle's data
# Verify start state will arrive stop state in Kaggle's train file.

import pandas as pd
import logging
from gamelib.conwaymap import ConwayMap
from gamelib.logutil import init_game_log, timing, kaggle_root


init_game_log('Kaggle data check')
side_len = 25
board_size = side_len * side_len
max_csv_rows = 1e6

def verify(full_scan_min_density):
    gl = ConwayMap(side_len, side_len, full_scan_min_density)

    data = pd.read_csv(kaggle_root + 'train.csv', index_col=0, dtype='str', nrows=max_csv_rows)
    for idx, row in data.iterrows():
        # Each row has values: index, delta, start * 625, stop * 625.
        delta = int(row[0])
        s_str = ''.join(row[1:(board_size+1)])
        s_bin = gl.str_to_bin(s_str)
        k_str = ''.join(row[(board_size+1):])
        k_bin = gl.str_to_bin(k_str)
        m_bin = gl.run(s_bin, iterations = int(delta))
        if not m_bin == k_bin:
            print('Failed to match game {}:'.format(idx))
            print('K: {}'.format(k_str))
            print('M: {}'.format(gl.bin_to_str(m_bin)))
            raise Exception('Verification failed. See earlier message.')


dens = [d * 0.1 for d in range(11)]
dens.extend([0.15, 0.18, 0.22, 0.25, 0.28, 0.32, 0.35])
dens = sorted(dens)
dens = [0.3]
timing()
time_record = []
for d in dens:
    verify(d)
    t = timing()
    print('Finished density {} in {}'.format(d, t))
    time_record.append(t)

rec = pd.DataFrame({'density': dens, 'time': time_record})
logging.info('\n{}'.format(rec))

