# Attempt to solve the reverse game by dense neural network.

import numpy as np
import pandas as pd
import time
from gamelib.reversenet import ReverseNet
from gamelib.logutil import init_game_log, kaggle_root


init_game_log('Reverse solver by neural net')

# Max number of rows from the train/test.csv files.
# Kaggle supplied training file has 50,001 lines
# Use small numbers to test first.
max_csv_rows = 60000

def timing(title):
    global prev_t
    t = time.time()
    t_sec = round(t - prev_t)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    prev_t = t
    print('{} used {}:{}:{}'.format(title, t_hour, t_min, t_sec))


def train_model():
    data = pd.read_csv(kaggle_root + 'train.csv', index_col=0, dtype='int', nrows=max_csv_rows)
    stop_states = data.iloc[:, :(board_size+1)]
    start_states = data.iloc[:, (board_size+1):]
    timing('Data loading')
    rn.train(stop_states, start_states)
    # On average, 15% of entries are 1 by this code
    # np.sum(np.array(start_states)) / np.prod(start_states.shape)



prev_t = time.time()
board = (25, 25)
board_size = np.product(board)
rn = ReverseNet(board_size, [256, 64, 32, 32, 32, 64], batches = 2048, epochs=100)

if rn.was_trained():
    rn.load()
else:
    train_model()
    rn.display_train()
timing('Modeling')


data = pd.read_csv(kaggle_root + 'test.csv', index_col=0, dtype='int', nrows=max_csv_rows)
prob, predict = rn.revert(data)
timing('Predict')


for delta in set(predict.delta):
    x = predict.loc[predict['delta']==delta,:]
    lives = np.sum(np.array(x.iloc[1:])==1)
    tot = len(np.array(x.iloc[1:]).flatten())
    print('Live cell weight for delta {} is {}.'.format(delta, lives / tot))

