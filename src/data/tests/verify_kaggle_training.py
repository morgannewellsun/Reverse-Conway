# Run this script from the dir Reverse-Conway/src.
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import time
from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn

def timing(msg):
    global prev_t
    t = time.time()
    t_sec = round(t - prev_t)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    prev_t = t
    print('{} - {}:{}:{}'.format(msg, t_hour, t_min, t_sec))



def verify_by_arr_rep():
    binary_prop = BinaryConwayForwardPropFn(numpy_mode=True)
    board_size = 25 * 25
    for idx, row in data.iterrows():
        delta = row[0]
        s_arr = row[1:(board_size+1)]
        k_arr = row[(board_size+1):]
        s_ten = np.array(s_arr).reshape((1, 25, 25, 1))
        k_ten = np.array(k_arr)
        m_ten = s_ten
        for _ in range(delta):
            m_ten = binary_prop(m_ten)
        model_res = m_ten.flatten().astype(int)
        if sum(abs(k_ten - model_res)) > 0:
            raise Exception('Failed to match game {}:'.format(idx))
    timing('Array representation')



max_csv_rows = 10
kaggle_root = '../../gamelife_data/kaggle/'
data = pd.read_csv(kaggle_root + 'train.csv', index_col=0, dtype='int', nrows=max_csv_rows)

prev_t = time.time()
verify_by_arr_rep()
