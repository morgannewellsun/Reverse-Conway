# Implementation of Conway's Game of Life.

import matplotlib.pyplot as plt
import numpy as np


class ConwayMap:
    
    def __init__(self, nrows, ncols, full_scan_density = 0.3):
        self.nrows = nrows
        self.ncols = ncols
        self.size = nrows * ncols
        # The minimum density to perform a full scan
        self._fs_min_dsty = full_scan_density
        self.is_still = False
        self._curr_state = 0
        self._prev_state = 0
        # This is a list of length self.size of numbers
        # Each number has binary representation (writen in three lines)
        # ... 111 ...
        # ... 101 ...
        # ... 111 ...
        self._pop_checker = list()
        # This is the list of cells impacted. The list has length self.size.
        # Each component is a set of 9 positions around its index.
        self._impacted = list()
        for r in range(nrows):
            for c in range(ncols):
                m11 = ((r-1)%nrows)*ncols + (c-1)%ncols
                m12 = ((r-1)%nrows)*ncols + c%ncols
                m13 = ((r-1)%nrows)*ncols + (c+1)%ncols
                m21 = (r%nrows)*ncols + (c-1)%ncols
                m22 = (r%nrows)*ncols + c%ncols
                m23 = (r%nrows)*ncols + (c+1)%ncols
                m31 = ((r+1)%nrows)*ncols + (c-1)%ncols
                m32 = ((r+1)%nrows)*ncols + c%ncols
                m33 = ((r+1)%nrows)*ncols + (c+1)%ncols
                self._pop_checker.append(
                    (1 << m11) + (1 << (m12)) + (1 << (m13))
                    + (1 << (m21)) + (1 << (m23))
                    + (1 << (m31)) + (1 << (m32)) + (1 << (m33))
                    )
                self._impacted.append({m11, m12, m13,
                                       m21, m22, m23,
                                       m31, m32, m33})


    def run(self, start_binary = None, iterations = 1):
        """ Arg start_binary is an integer whose binary representation
        specifies the locations of live cells.
        If this argument is not specified, the current state is assumed.
        For example, consider a 3X3 board.
        The base (0,0) is at the lower right corner.
        Index increases from right to left, and from bottom to top.
        State 393 = 0b110001001.
        110 001 001 represent living cells at
        (0,0), (1,0), (2,1), (2,2). 
        Return the state (an integer) after iterations.
        """
        if start_binary:
            self._curr_state = start_binary
        for j in range(iterations):
            self._prev_state = self._curr_state
            self._advance()
            if self._curr_state == self._prev_state:
                self.is_still = True
                break
        return self._curr_state


    def _advance(self):
        # Check if each cell will be live in the next generation.
        if bin(self._curr_state).count('1') < self.size * self._fs_min_dsty:
            # The game board is not densely populated.
            # We select cells to check status for the next generation.
            # This is the set of all positions possibly having live cells.
            selected = set()
            for j, s in enumerate(bin(self._curr_state)[:1:-1]):
                if s == '1':
                    selected |= self._impacted[j]
            self._curr_state = sum(self._cell_status(p) for p in selected)
        else:
            # A full scan of all cells in the game board.
            self._curr_state = sum(self._cell_status(p) for p in range(self.size))


    def _cell_status(self, pos):
        neighbors = bin(self._curr_state & self._pop_checker[pos]).count('1')
        if neighbors < 2:
            return 0
        if neighbors == 2:
            return self._curr_state & (1 << pos)
        if neighbors == 3:
            return 1 << pos
        return 0

    
    def str_to_bin(self, s):
        """ Convert string representation of a game board into an integer.
        The string only contains chars either '0' or '1'.
        The base point is the lower right corner.
        Example, s = '000100001' in a 3x3 board has live cells at
        (1, 0) and (1, 2).
        """
        return int(s, 2)


    def bin_to_str(self, chromo):
        lower = bin(chromo)[2:]
        return '0'*(self.size - len(lower)) + lower


    def bin_to_2d(self, chromo):
        arr = [int(c) for c in self.bin_to_str(chromo)]
        return np.array(arr).reshape((self.nrows, self.ncols)).tolist()


    def animate(self, start_binary, iterations, rest = 0.1):
        plt.axis('off')
        plt.title('Game of Life')
        plt.ion()
        board = self.bin_to_2d(start_binary)
        im = plt.imshow(board, cmap = plt.cm.plasma)
        self._curr_state = start_binary
        for iter in range(iterations):
            plt.pause(rest)
            board = self.bin_to_2d(self.run())
            im.set_data(board)
            plt.title('Game of Life: {}'.format(iter+1))
        plt.pause(3)
