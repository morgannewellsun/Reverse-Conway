import datetime as dt
import os

import numpy as np
import tensorflow as tf

from components.true_target_acc_fn import TrueTargetAccFn
from components.true_target_loss_fn import TrueTargetLossFn
from data.kaggle_data_generator import KaggleDataGenerator
from models.baseline_conv_model import BaselineConvModel


class TrainingRunner:

    def __init__(self):
        pass

    def run(self, delta_steps: int):

        # prepare data generator
        train_generator = KaggleDataGenerator(batch_size=128, batches_per_epoch=100, delta_steps=delta_steps)

        # prepare and compile the model
        reverse_model = BaselineConvModel(n_filters=256, n_hidden_layers=12)
        loss_fn = TrueTargetLossFn(delta_steps=delta_steps)
        acc_fn = TrueTargetAccFn(delta_steps=delta_steps)
        reverse_model.compile(optimizer="adam", loss=loss_fn, metrics=[acc_fn])

        # run training loop
        root_output_dir = r"D:\Documents\Reverse Conway\Output"
        run_folder_name = dt.datetime.now().strftime("%y%m%d%H%M%S") + f" (delta {delta_steps})"
        run_folder_name = os.path.join(run_folder_name, "")
        run_output_dir = os.path.join(root_output_dir, run_folder_name)
        os.mkdir(run_output_dir)
        checkpoint_filepath = os.path.join(run_output_dir, "best_checkpoint.hdf5")
        callbacks = [tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, monitor="TrueTargetAcc", verbose=1, save_best_only=True)]
        reverse_model.fit(x=train_generator, epochs=10000, callbacks=callbacks)


if __name__ == "__main__":
    runner = TrainingRunner()
    runner.run(delta_steps=4)




