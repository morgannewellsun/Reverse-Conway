import pandas as pd
import time
from components.conwaymap import ConwayMap

# Randomly take a row in the data and verify the numbers are correct.
def sample_verify():
    conway = ConwayMap(nrows=25, ncols=25)
    nrows = len(data)
    # Randomly take a row to verify.  Avoid random() since seed is fixed.
    # Use current time digits after the decimal point.
    j = int(str(time.time()%1)[2:])%nrows
    row = data.iloc[j, :]
    (game_index, delta, target_lives, cnn_lives, cnn_errors,
     ga_lives, ga_errors) = map(int, row[:7])
    (target, cnn, ga) = row[7:]
    end_state = int(target)
    cnn_state = int(cnn)
    ga_state = int(ga)
    
    expect = bin(end_state).count('1')
    if not expect == target_lives:
        raise Exception('Game {} failed target_live {} vs expected {}'.format(
            game_index, target_lives, expect))
        
    expect = bin(cnn_state).count('1')
    if not cnn_lives == expect:
        raise Exception('Game {} failed cnn_lives {} vs expected {}'.format(
            game_index, cnn_lives, expect))
        
    expect = bin(conway.run(cnn_state, delta)^end_state).count('1')
    if not cnn_errors == expect:
        raise Exception('Game {} failed cnn_errors {} vs expected {}'.format(
            game_index, cnn_errors, expect))
        
    expect = bin(ga_state).count('1')
    if not ga_lives == expect:
        raise Exception('Game {} failed ga_lives {} vs expected {}'.format(
            game_index, ga_lives, expect))
        
    expect = bin(conway.run(ga_state, delta)^end_state).count('1')
    if not ga_errors == expect:
        raise Exception('Game {} failed ga_errors {} vs expected {}'.format(
            game_index, ga_errors, expect))



output_dir = '../../gamelife_data/output/'
data = pd.read_csv(output_dir + 'results.csv', index_col=0)
sample_verify()

game_size = 25 * 25
# statistics by errors
err_col = ['delta ' + str(j) for j in range(6)]
err_stats = pd.DataFrame([[0]*6]*game_size, columns=err_col)
# statistics by number of lives at the end state
liv_stats = pd.DataFrame([[0]*3]*game_size, columns=(
    'count', 'cnn_fails', 'ga_fails'))
del_stats = pd.DataFrame([[0]*5]*6, columns=(
    'count', 'cnn_hits', 'ga_hits', 'cnn_fails', 'ga_fails'))

for j, row in data.iterrows():
    (game_index, delta, target_lives, cnn_lives, cnn_errors,
     ga_lives, ga_errors) = map(int, row[:7])
    # game_index = row[0]
    # delta = row[1]
    # target_lives = row[2]
    # cnn_lives = row[3]
    # cnn_errors = row[4]
    # ga_lives = row[5]
    # ga_errors = row[6]
    
    err_stats.iloc[ga_errors, delta] += 1
    liv_stats.iloc[target_lives, 0] += 1
    liv_stats.iloc[target_lives, 1] += cnn_errors
    liv_stats.iloc[target_lives, 2] += ga_errors
    del_stats.iloc[delta, 0] += 1
    del_stats.iloc[delta, 3] += cnn_errors
    del_stats.iloc[delta, 4] += ga_errors
    if cnn_errors == 0:
        del_stats.iloc[delta, 1] += 1
    if ga_errors == 0:
        del_stats.iloc[delta, 2] += 1

err_stats['total'] = err_stats.sum(axis=1)
err_stats = err_stats.loc[err_stats['total']>0, :]
err_stats.to_csv(output_dir + 'err_stats.csv')

liv_stats = liv_stats.loc[liv_stats['count']>0, :]
liv_stats['cnn_accuracy'] = 1 - liv_stats['cnn_fails'] / liv_stats['count'] / game_size
liv_stats['ga_accuracy'] = 1 - liv_stats['ga_fails'] / liv_stats['count'] / game_size
liv_stats.to_csv(output_dir + 'liv_stats.csv')

del_stats = del_stats[del_stats.index > 0]
del_stats['cnn_accuracy'] = 1 - del_stats['cnn_fails'] / del_stats['count'] / game_size
del_stats['ga_accuracy'] = 1 - del_stats['ga_fails'] / del_stats['count'] / game_size
del_stats.to_csv(output_dir + 'del_stats.csv')
