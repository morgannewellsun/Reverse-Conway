# Implementation of Conway's Game of Life.

import matplotlib.pyplot as plt
import numpy as np


class GameLife:
    
    def __init__(self, nrows, ncols):
        self._life_min = 2
        self._life_max = 3
        self._life_rev = 3
        self._nrows = nrows
        self._ncols = ncols
        self.evolutions = None
        self.is_still = False
        self._curr_state = None
        self._prev_state = None
        self._procssed_rows = None     # The row that has been processed.
    
    
    def run(self, start_state = None, start_array = None, iterations = 100):
        """ Arg start_state is a list of locations of live cells.
        A location is a tuple of two integers no less than 0.
        Return True if the board is still, False if the board changes.
        """
        
        if start_state is None:
            if not start_array is None:
                start_state = self._array_to_state(start_array)
            else:
                raise Exception('Initial state or initial board must be provided.')
        
        # A list of dictionaries, each maps row number
        # to a set of living cell column numbers.
        self.evolutions = list()
        # This only records those living cells.
        # The key is row number, value is a set of columne numbers.
        self._prev_state = start_state.copy()
        self.evolutions.append(self._prev_state)
        self.is_still = False

        for j in range(iterations):
            self._curr_state = dict()
            self._one_step()
            if self._curr_state == self._prev_state:
                # print('Still after step {}'.format(j))
                self.is_still = True
                break
            self._prev_state = self._curr_state.copy()
            self.evolutions.append(self._prev_state)
            
        return self.is_still
    
    
    def last_board(self):
        return self._state_to_board(self.evolutions[-1])


    def last_array(self):
        return self._state_to_array(self.evolutions[-1])


    def _state_to_board(self, lives):
        """ Convert a life dictionary to 2D array. 
        """
        arr = [[0] * self._ncols for j in range(self._nrows)]
        for r, cset in lives.items():
            for c in cset:
                arr[r][c] = 1
        return arr


    def _state_to_array(self, lives):
        """ Convert a life dictionary to 1D array. 
        """
        arr = [0] * (self._nrows * self._ncols)
        for r, cset in lives.items():
            for c in cset:
                arr[r * self._nrows + c] = 1
        return arr
    
    
    def _array_to_state(self, arr):
        state = dict()
        for r in range(self._nrows):
            for c in range(self._ncols):
                if arr[r * self._ncols + c]:
                    state.setdefault(r, set()).add(c)
        # map(lambda j: state.setdefault(j // self._ncols, set()).add(j % self._ncols) if arr[j] else None, arr)
        return state


    def animate(self, rest = 0.1):
        plt.axis('off')
        plt.title('Game of Life')
        plt.ion()
        board = self._state_to_board(self.evolutions[0])
        im = plt.imshow(board, cmap = plt.cm.plasma)
        iter = 0
        for lives in self.evolutions:
            plt.pause(rest)
            board = self._state_to_board(lives)
            # print(board)
            im.set_data(board)
            plt.title('Game of Life: {}'.format(iter))
            iter += 1
        plt.pause(3)


    def _should_live(self, p, row1, row2, row3):
        # row1, row2, and row3 are sets of live cells in three rows.
        # Rows row1 and row3 are two rows above and below row2.
        # We are checing if cell p in row2 should be live.

        xnbor = {(p-1) % self._ncols, p, (p+1) % self._ncols}
        cnt = len(xnbor & row1) + len(xnbor & row2) + len(xnbor & row3)
        if p in row2:
            # Live cell case. There are cnt-1 live neighbors.
            if cnt > self._life_min and cnt < self._life_max + 2:
                return True
            else:
                return False
        else:
            if cnt == self._life_rev:
                return True
        return False


    def _one_row(self, r):
        r = r % self._nrows
        if r in self._procssed_rows:
            return
        self._procssed_rows.add(r)

        row1 = self._prev_state.get((r - 1) % self._nrows, set())
        row2 = self._prev_state.get(r, set())
        row3 = self._prev_state.get((r + 1) % self._nrows, set())
        if len(row1) + len(row2) + len(row3) < self._life_min:
            # The 3 neighbor rows don't contain enough live cells.
            # All cells in this row will be dead.
            return
        
        # The column indexes of live cells in row r in next life iteration.
        lifes = set()
        processed_cells = set()
        for c in row1 | row2 | row3:
            for p in (c-1, c, c+1):
                q = p % self._ncols
                if not q in processed_cells:
                    if self._should_live(q, row1, row2, row3):
                        lifes.add(q)
                    processed_cells.add(q)

        self._curr_state[r] = lifes


    def _one_step(self):
        # Go forward to the next life iteration.
        self._procssed_rows = set()
        for r in self._prev_state.keys():
            self._one_row(r - 1)
            self._one_row(r)
            self._one_row(r + 1)


    def visual_error(self, start_array, iterations, end_array):
        """ Visually show diff between projected end array from given start
        and the target end.
        """
        self.run(start_array=start_array, iterations=iterations)
        shp = (self._nrows, self._ncols)
        plt.axis('off')
        plt.title('Game of Life comparison for delta {}'.format(iterations))
        board_start = np.array(start_array).reshape(shp)
        board_calc = np.array(self._state_to_board(self.evolutions[-1]))
        board_true = np.array(end_array).reshape(shp)
        board_diff = board_calc ^ board_true
        fill = -1
        big_board = np.full((2*self._nrows+1, 2*self._ncols+1), fill)
        big_board[0:self._nrows, 0:self._ncols] = board_start
        big_board[0:self._nrows, (self._ncols+1):] = board_diff
        big_board[(self._ncols+1):, 0:self._ncols] = board_calc
        big_board[(self._ncols+1):, (self._ncols+1):] = board_true
        plt.imshow(big_board, cmap = plt.cm.plasma)

