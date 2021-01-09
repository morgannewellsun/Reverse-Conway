### Overview

This repository contains our solution to [the 2020 Conway's Reverse Game of Life Kaggle competition](https://www.kaggle.com/c/conways-reverse-game-of-life-2020).

The objective is to create an algorithm which, given a [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) board (the target board) and a number of timesteps (`delta`), can predict a board which, when propagated forwards `delta` timesteps according to the rules of Conway's Game of Life, results in the original input (target) board. In other words, the algorithm must "reverse time" by `delta` steps.

### Solution Highlights

- CNN models with specialized architecture trained using a dynamic blend of two loss functions to predict `delta` timesteps backwards in time.
    * **For much more discussion regarding the training of these CNN models, see the `README.md` in the [`src` folder](https://github.com/morgannewellsun/Reverse-Conway/tree/master/src).**
- During inference, CNN predictions are refined using a genetic algorithm implemented in tensorflow.
    * **For much more discussion regarding the refinement of CNN predictions using a genetic algorithm, see the `README.md` in the in the [`kaggle` folder](https://github.com/morgannewellsun/Reverse-Conway/tree/master/kaggle).**

### Dependencies

- Python (3.7 tested)
- numpy
- pandas
- tensorflow (2.1 and 2.3 tested, data generation bug in 2.1)
- matplotlib

