### <u>Kaggle Submission Files</u>

This folder contains the files and folder structure necessary for submission to Kaggle.

### Folder Overview

The `Reverse-Conway/input` folder contains copies of pretrained moels found in `Reverse-Conway/pretrained_models`.
It is also used by the Kaggle framework as the location of evaluation data.

The `Reverse-Conway/working` folder contains notebooks/scripts. 
The primary file of interest here is `Reverse-Conway/working/conway_parallel.py`, which was the script submitted to Kaggle.

### Strategy Overview

`conway_parallel.py` is a script that can be run independently.
It does not depend on code from other parts of the project.
It only depends on the pretrained models placed in `Reverse-Conway/input` folder.

The general strategy employed in this script is as follows:

1. Partition the set of evaluation problems according to their `delta` value (the required number of backwards timesteps from the provided final state).
2. For problems with a `delta` value of `1`, `2`, or `3`, use the "stackwise" solver method.
3. For problems with a `delta` value of `4` or `5`, use the "oneshot" solver method.

Both the "stackwise" and "oneshot" solver methods employ both 
convolutional neural network (CNN) and genetic algorithm (GA), but in different ways. 
Both the CNN and GA are implemented in Tensorflow in order to improve inference time, 
allowing more GA iterations within the time limit and thus improved performance.


### Solver Subroutines

CNN usage in the solver algorithms is very straightforward.
In all cases, a CNN model trained to revert a fixed number of timesteps is provided to the solver,
and the solver simply treats the model as a black-box function. For more information about the training 
of these CNN models, see [the src folder](https://github.com/morgannewellsun/Reverse-Conway/tree/master/src),
which contains implementations for model architecture, data generation, and training.

There are two key GA-related subroutines used in both solvers.
For simplicity, vectorization is not considered in the pseudocode below.

```
GA-INITALIZE-POPULATION

INPUT:
- cell_probabilities:   (25 by 25) grid of floats representing predicted cell-alive probabilities

PARAMETERS:
- ga_n_dynamic:         integer
- ga_n_static:          integer
- ga_static_span:       float

OUTPUT:
- List of (25 by 25) grids of booleans representing the initial GA population.

1.  ga_population <- []
2.  for ga_n_dynamic iterations:
3.      ga_population <- ga_population + [sample (25 by 25) grid of random booleans, weighted by cell_probabilities]
4.  static_thresholds <- [ga_n_static uniformly spaced floats between 0.5-ga_static_span and 0.5+ga_static_span]
5.  for each static_threshold in static_thresholds:
6.      ga_population <- ga_population + [cell_probabilities >= static_threshold]
7.  return ga_population
```

```
GA-ITERATE

INPUT:
- ga_population:        List of (25 by 25) grids of booleans representing the GA population.

PARAMETERS:
- ga_mutation_rate:     float

OUTPUT:
- List of (25 by 25) grids of booleans representing the GA population.

1.  n_original_population <- len(ga_population)
2.  ga_population_shuffled <- shuffle ga_population
3.  ga_population_mutations <- []
4.  for each board in ga_population:
5.      mutator <- (25 by 25) grid of random booleans, weighted by ga_mutation_rate
6.      mutation_result_board <- board ^ mutator
7.      ga_population_mutations <- ga_population_mutations + [mutation_result_board]
8.  ga_population_crossovers <- []
9.  for integer i in len(ga_population):
10.     swapper <- (25 by 25) grid of 50/50 random booleans
11.     crossover_result_board <- (ga_population[i]) & swapper | (ga_population_shuffled[i] & ~swapper)
12.     ga_population_crossovers <- ga_population_crossovers + [crossover_result_board]
13. ga_population <- ga_population + ga_population_mutations + ga_population_crossovers
14. ga_population <- select best n_original_population boards in ga_population
15. return ga_population
```

### Stackwise Solver

The stackwise solver iterates backwards one timestep at a time.
This solver requires a CNN trained to predict one timestep backwards in time.
For each timestep, the CNN is used to generate a (25 by 25) grid of alive-probabilities, which the GA then iterates on.
For simplicity, vectorization is not considered in the pseudocode below.

```
SOLVER-STACKWISE

INPUT:
- target_state:         (25 by 25) boolean grid representing target board state
- cnn_model:            CNN model trained to revert a given board state by one timestep
- delta:                integer number of timesteps to revert

PARAMETERS:
- ga_iterations:        integer
 
OUTPUT:
- (25 by 25) boolean grid representing predicted starting board state

1.  for delta iterations:
2.      cell_probabilities <- cnn_model(target_state)
3.      ga_population <- GA-INITALIZE-POPULATION(cell_probabilities)
4.      for ga_iterations iterations:
5.          ga_population <- GA-ITERATE(ga_population)
6.      pred_board <- select best board from ga_population
7.  return pred_board
```

### Oneshot Solver

The oneshot solver iterates backwards all `delta` timesteps at once.
This solver requires a CNN trained to predict `delta` timesteps backwards in time.
Once again, for simplicity, vectorization is not considered in the pseudocode below.

```
SOLVER-ONESHOT

INPUT:
- target_state:         (25 by 25) boolean grid representing target board state
- cnn_model:            CNN model trained to revert a given board state by delta timesteps
- delta:                integer number of timesteps to revert

PARAMETERS:
- ga_iterations:        integer
 
OUTPUT:
- (25 by 25) boolean grid representing predicted starting board state

1.  cell_probabilities <- cnn_model(target_state)
2.  ga_population <- GA-INITALIZE-POPULATION(cell_probabilities)
3.  for ga_iterations iterations:
4.      ga_population <- GA-ITERATE(ga_population)
5.  pred_board <- select best board from ga_population
6.  return pred_board
```

