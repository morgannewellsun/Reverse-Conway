import math
import numpy as np

class CnnMan:
    # A CNN solver manager. It produces solutions to Conway games using CNN.
    
    def __init__(self, cnn_reverters, stepwise):
        # Arg cnn_reverters is a dictionary from int (delta)
        # to a list of CNN solvers. Each solver accepts array of np.float32.
        self._cnn_rerverters = cnn_reverters
        # Make sure each delta has a value for later ease of manipulations.
        for j in range(1, 6):
            if not j in self._cnn_rerverters:
                self._cnn_rerverters[j] = []
        self._stepwise = stepwise


    def _revert_many(self, model, stop_states):
        # Use CNN to revert many boards.
        # Arg stop_states is an array of size
        # (number of boards, nrows, ncols, 1).
        cnn_result = model(stop_states.astype(np.float32)).numpy()
        return cnn_result >= 0.5


    def _revert_one(self, model, stop_state, popsize):
        # Use CNN to revert a single board.
        # Arg popsize is the number of output game boards.
        # Arg stop_state is array of size (1, nrows, ncols, 1).
        cnn_result = model(stop_state).numpy()
        sorted_probs = sorted(cnn_result.flatten())
        life50 = (cnn_result < 0.5).sum()
        half_pop = int(popsize / 2)
        if life50 - half_pop < 0:
            selected = range(popsize)
        elif life50 - half_pop + popsize < np.prod(stop_state.shape):
            selected = range(life50 - half_pop, life50 - half_pop + popsize)
        else:
            selected = range(-popsize, 0)
        # This is a list of 1D 0/1 arrays representing the boards from CNN.
        return np.array([(cnn_result[0] > sorted_probs[j]) for j in selected])

    
    def revert(self, stop_state, delta, popsize):
        # Return initial game boards as an array of bool with size
        # (count, width, height, 1), with count at least popsize.
        
        # This is CNN model to revert in one shot.
        model_d = self._cnn_rerverters[delta]
        if self._stepwise and delta > 1:
            # This uses CNN 1-step revert model iteratively.
            model_1 = self._cnn_rerverters[1]
        else:
            # Don't revert by iterative 1-step model.
            model_1 = []
        tot_mods = len(model_d) + len(model_1)
        if tot_mods == 0:
            # No CNN models available, use random array.
            print('Random game boards are used.')
            sz = list(stop_state.shape)
            sz[0] = popsize
            return np.random.randint(2, size=sz).astype(bool)
        
        unit_size = math.ceil(popsize / tot_mods)
        cnn_results = []
        for m in model_d:
            # Revert delta steps back in one shot.
            model_res = self._revert_one(m, stop_state, unit_size)
            cnn_results.extend(model_res)
        for m in model_1:
            # Revert 1 step in each iteration.
            res = self._revert_one(m, stop_state, unit_size)
            for _ in range(delta - 1):
                res = self._revert_many(m, res)
            cnn_results.extend(res)
        return np.array(cnn_results)
