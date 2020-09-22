import datetime as dt
import os

import numpy as np
import tensorflow as tf

from components.prob_conway_forward_prop import ProbConwayForwardProp
from data.baseline_data_generator import BaselineDataGenerator
from models.baseline_conv_model import BaselineConvModel


class TrainingRunner:

    def __init__(self):
        pass

    def run(self, n_steps: int):

        # prepare data generator
        train_generator = BaselineDataGenerator(batch_size=128)
        # val_generator = BaselineDataGenerator(batch_size=128, batches_per_epoch=10)

        # prepare and compile the model
        reverse_model = BaselineConvModel(n_filters=256, n_hidden_layers=12)
        forward_layer = ProbConwayForwardProp()
        reverse_then_forward = [reverse_model]
        for _ in range(n_steps):
            reverse_then_forward.append(forward_layer)
        reverse_then_forward = tf.keras.Sequential(reverse_then_forward)
        reverse_then_forward.compile(optimizer="adam", loss="binary_crossentropy")

        # run training loop
        # root_output_dir = r"D:\Documents\Reverse-Conway\Output"
        # run_output_dir = os.path.join(root_output_dir, dt.datetime.now().strftime("%y%m%d%H%M%S"))
        # os.mkdir(run_output_dir)
        # checkpoint_filepath = os.path.join(run_output_dir, "best_checkpoint.hdf5")
        # callbacks = [tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, verbose=1, save_best_only=True)]
        reverse_then_forward.fit(x=train_generator, epochs=10000)


if __name__ == "__main__":
    runner = TrainingRunner()
    runner.run(1)




