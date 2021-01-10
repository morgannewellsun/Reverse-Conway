### <u>Model Training Framework</u>

This folder contains the code framework used to train CNN models used to reverse
given board states by a fixed number of timesteps.
For more information about the integration of these models in the solution strategy,
see [the kaggle folder](https://github.com/morgannewellsun/Reverse-Conway/tree/master/kaggle).

### Folder Overview

- The `Reverse-Conway/src/framework` folder contains various runner scripts for different
training methods, as well as some visualization code.
- The `Reverse-Conway/src/components` folder contains a shared library of Tensorflow
functions relevant to Conway's Game of Life, as well as corresponding unittests.
- The `Reverse-Conway/src/data` folder contains various training data generators.
- The `Reverse-Conway/src/models` folder contains definitions of various Keras
model architectures which were experimented with.

### Training Methodologies

In all training methodologies, the model is given a `(25, 25)` grid of booleans representing
a target board state as input. The model is expected to output a `(25, 25)` grid of floats
between `0` and `1`, representing predicted cell-alive probabilities for the board `delta`
timesteps prior. A different set of weights must be trained for each value of `delta`.

Three different training methodologies were used: unsupervised, supervised, and crossfade.

*NOTE: Admittedly, the terminology "supervised" and "unsupervised" are not used in same sense
as they are in most other machine learning literature, 
but it wouldn't really be worth renaming things now.
In the context of this project, please forget about the conventional machine learning
definitions of these words.*

**"Unsupervised" training:**

The idea for this training scheme is from 
[this Kaggle notebook](https://www.kaggle.com/akashsuper2000/crgl-probability-extension-true-target-problem).

In this training methodology model, the "true target loss function" is used, 
which is implemented in `components/true_target_loss_fn.py`. 
The loss function takes a given grid of alive-probabilities representing the predicted starting board state, 
probabilistically propagates it forward a `delta` steps according to the rules of Conway's Game of Life, 
then computes the binary crossentropy between the result and a given binary target state.

The advantage of this training method lies in the fact that the "true target loss function"
incorporates a differentiable version of the rules of Conway's Game of Life. This allows
the model to be trained to yield starting-state predictions which, 
after being propagated forward `delta` steps, match the target state very closely.

However, using this training scheme, the model is not constrained or even incentivized
to predict starting board states that are likely to arise naturally during Conway's Game of Life. 
The model often outputs predictions containing extremely "unrealistic" features, 
such as large swathes of living cells, most of which immediately die the following timestep.

**"Supervised" training:**

In this training methodology, the binary crossentropy between the model's prediction
and the starting-state that yielded the input target state is used as the loss.

The advantage to this training method is that the model is incentivized to predict
"realistic" starting-states, unlike in the "unsupervised" training methodology.

However, the disadvantage is that the model is not directly incentivized to ensure that
the predicted starting-states will, after being propagated forward `delta` steps,
will resemble the input target.

**"Crossfade" training:**

The loss function used is implemented in `components/crossfade_loss_fn.py`. 
This loss function gradually transitions from the "supervised" loss function (i.e. BCE) 
to a 50/50 blend of the "unsupervised" and "supervised" loss functions.

Intuitively, this seems to result in the model's weights settling into an optimum area
where predictions are fairly realistic thanks to the initial use of the "supervised" methodology,
before being "refined" via the "unsupervised" methodology. 

The idea of this loss function was more or less the biggest breakthrough during the course of this project!

### Model Architectures

`components/roll_padding_2d_layer.py` contains a tensorflow function which pads a tensor on each side
by rolling the opposite side over. For this competition, the Conway game board is a torus,
that is, the bottom row is "above" the top row of the board, and the leftmost column is
"to the right of" the rightmost column. This "roll padding" layer allows this torus geometry
to be incorporated into the architecture of the CNN models. 
All models we implemented exclusively use this padding.

Various model architectures were tested, including:

- Basic baseline model with stacked (3 by 3) convolution layers.
- CNN model with repeating blocks of [(3 by 3), (1 by 1), (1 by 1) ...] convolutional layers. 
- A miniature version of Densenet.
- A miniature version of ResNeXt.

Ultimately, the baseline model proved perform the best given the GPU memory constraints 
(training was performed on a desktop computer with a GTX1080Ti graphics card and 16GB of RAM).

### Training and Test Data

All training and test data is randomly generated on-the-fly using generator functions.
Because there are an astronomical number of possible board states, randomly generating data
on-the-fly makes overfitting far less likely.
The data generation methodology is an exact copy of [the test data generation
methodology specified by the competition](https://www.kaggle.com/c/conways-reverse-game-of-life-2020/data).

Note that a [tensorflow 2.1 bug](https://github.com/tensorflow/tensorflow/issues/35911) 
causes `tf.keras.utils.Sequence` generators to not shuffle/regenerate data every epoch.
A rather hacky workaround involves overriding the `__len__` function such that it calls `on_epoch_end`.
Alternatively, one can upgrade to tensorflow 2.3.

### Misc

- `components/binary_conway_forward_prop_fn.py` and `components/prob_conway_forward_prop_fn.py`
contain tensorflow functions implementing the logic for propagating a given board state forward in time by one step.
The binary version accepts arrays of boolean living/dead states, 
while the probabilistic version accepts arrays of living probabilities. 
The probabilistic version is differentiable,
and thus allows the game logic to be incorporated into loss functions.
- `components/prob_conway_to_binary_conway_fn.py` contains a tensorflow function to convert
an array of living probabilities to an array of boolean living/dead states.
- `framework/visualization_callback.py` and `framework/visualizer.py` contain board visualization code.
