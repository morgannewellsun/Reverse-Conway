import pandas as pd
import numpy as np
import time
from components.binary_conway_forward_prop_fn import BinaryConwayForwardPropFn

# Randomly take a row in the data and verify the numbers are correct.
def sample_verify(data):
    conway = BinaryConwayForwardPropFn(numpy_mode=True)
    nrows = len(data)
    # Randomly take a row to verify.  Avoid random() since seed is fixed.
    # Use current time digits after the decimal point.
    j = int(str(time.time()%1)[2:])%nrows
    row = data.iloc[j, :]
    (game_index, delta, target_lives,
     ga_lives, ga_errors) = map(int, row[:5])
    (target, ga) = row[5:]

    end_state = np.array(list(target)).astype(int).reshape((1,25,25,1))
    expect =end_state.sum()
    if not expect == target_lives:
        raise Exception('Game {} failed target_live {} vs expected {}'.format(
            game_index, target_lives, expect))

    ga_state = np.array(list(ga)).astype(int).reshape((1,25,25,1))
    expect = ga_state.sum()
    if not ga_lives == expect:
        raise Exception('Game {} failed ga_lives {} vs expected {}'.format(
            game_index, ga_lives, expect))
    expect = abs(conway(ga_state, delta) - end_state).sum()
    if not ga_errors == expect:
        raise Exception('Game {} failed ga_errors {} vs expected {}'.format(
            game_index, ga_errors, expect))

    print('Verified row {} delta {} on game {}.'.format(j, delta, game_index))



def post_run_report(output_dir):
    data = pd.read_csv(output_dir + 'results.csv', index_col=0)
    sample_verify(data)
    
    game_size = 25 * 25
    # statistics by errors
    err_col = ['delta ' + str(j) for j in range(6)]
    err_stats = pd.DataFrame([[0]*6]*game_size, columns=err_col)
    # statistics by number of lives at the end state
    liv_stats = pd.DataFrame([[0]*3]*game_size, columns=(
        'count', 'game_hits', 'cell_fails'))
    del_stats = pd.DataFrame([[0]*3]*6, columns=(
        'count', 'game_hits', 'cell_fails'))
    
    for j, row in data.iterrows():
        (game_index, delta, target_lives,
         ga_lives, ga_errors) = map(int, row[:5])
    
        err_stats.iloc[ga_errors, delta] += 1
        liv_stats.iloc[target_lives, 0] += 1
        liv_stats.iloc[target_lives, 2] += ga_errors
        del_stats.iloc[delta, 0] += 1
        del_stats.iloc[delta, 2] += ga_errors
        if ga_errors == 0:
            liv_stats.iloc[target_lives, 1] += 1
            del_stats.iloc[delta, 1] += 1

    err_stats['total'] = err_stats.sum(axis=1)
    err_stats = err_stats.loc[err_stats['total']>0, :]
    err_stats.to_csv(output_dir + 'err_stats.csv')
    
    liv_stats = liv_stats.loc[liv_stats['count']>0, :]
    liv_stats['accuracy'] = 1 - liv_stats['cell_fails'] / liv_stats['count'] / game_size
    liv_stats.to_csv(output_dir + 'liv_stats.csv')
    
    del_stats = del_stats[del_stats.index > 0]
    del_stats['accuracy'] = 1 - del_stats['cell_fails'] / del_stats['count'] / game_size
    del_stats.to_csv(output_dir + 'del_stats.csv')


if __name__ == "__main__":
    # This needs to be run at Reverse-Conway/src
    output_dir = '../../gamelife_data/output/'
    post_run_report(output_dir)
