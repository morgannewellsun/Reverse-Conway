import datetime as dt
import json
import os
import time
from typing import *

import tensorflow as tf

from components.true_target_acc_fn import TrueTargetAccFn
from components.true_target_loss_fn import TrueTargetLossFn
from data.pretty_test_target import pretty_test_target
from data.kaggle_supervised_data_generator import KaggleSupervisedDataGenerator
from data.kaggle_unsupervised_data_generator import KaggleUnsupervisedDataGenerator
from data.kaggle_supervised_delta_one_data_generator import KaggleSupervisedDeltaOneDataGenerator
from framework.visualization_callback import VisualizationCallback
from models.baseline_conv_model import BaselineConvModel
from models.simplified_densenet_model import SimplifiedDenseNetModel
from models.simplified_resnext_model import SimplifiedResNeXtModel
from models.three_plus_ones_block_model import ThreePlusOnesBlockModel


class TrainingRunnerTransition:

    @staticmethod
    def run(
            *,
            root_output_dir: str,
            delta_steps: int,
            train_generator_name: str,
            train_generator_config: dict,
            val_generator_name: Optional[str] = None,
            val_generator_config: Optional[dict] = None,
            val_freq: int = 1,
            model_name: str,
            model_config: dict,
            early_stop_patience: int,
            early_stop_min_delta: float,
            max_epochs: int,
            visualize_first_n: int
    ) -> None:

        # create run folder and save run config
        run_config_dict = locals().copy()
        run_folder_name = dt.datetime.now().strftime("%Y%m%dT%H%M%S") + f" (delta {str(delta_steps)})"
        run_output_dir_complete = os.path.join(root_output_dir, run_folder_name, "")
        run_output_dir = os.path.join(root_output_dir, run_folder_name + " (incomplete)", "")
        os.mkdir(run_output_dir)
        run_config_filepath = os.path.join(run_output_dir, "run_config.json")
        with open(run_config_filepath, "w") as file:
            json.dump(run_config_dict, file, indent=4)

        # prepare training data generator
        if train_generator_name == "KaggleSupervisedDataGenerator":
            train_generator = KaggleSupervisedDataGenerator(delta_steps=delta_steps, **train_generator_config)
        elif train_generator_name == "KaggleSupervisedDeltaOneDataGenerator":
            train_generator = KaggleSupervisedDeltaOneDataGenerator(**train_generator_config)
        else:
            raise ValueError("Invalid supervised data generator specified.")

        # prepare validation data generator
        if val_generator_name is not None:
            if val_generator_name == "KaggleSupervisedDataGenerator":
                val_generator = KaggleSupervisedDataGenerator(delta_steps=delta_steps, **val_generator_config)
            elif val_generator_name == "KaggleSupervisedDeltaOneDataGenerator":
                val_generator = KaggleSupervisedDeltaOneDataGenerator(**val_generator_config)
            else:
                raise ValueError("Invalid supervised data generator specified.")
        else:
            val_generator = None

        # prepare the model
        if model_name == "BaselineConvModel":
            reverse_model = BaselineConvModel(**model_config)
        elif model_name == "ThreePlusOnesBlockModel":
            reverse_model = ThreePlusOnesBlockModel(**model_config)
        elif model_name == "SimplifiedDenseNetModel":
            reverse_model = SimplifiedDenseNetModel(**model_config)
        elif model_name == "SimplifiedResNeXtModel":
            reverse_model = SimplifiedResNeXtModel(**model_config)
        else:
            raise ValueError(f"{model_name} is not a supported model.")
        supervised_loss_fn = tf.keras.losses.BinaryCrossentropy()  # bce loss monitored
        acc_fns = [
            TrueTargetAccFn(delta_steps=0, name="StartAcc"),
            TrueTargetAccFn(delta_steps=delta_steps, name="StopAcc", y_true_is_start=True)]
        reverse_model.compile(optimizer="adam", loss=supervised_loss_fn, metrics=acc_fns)
        reverse_model.build(input_shape=(None, 25, 25, 1))
        reverse_model.summary()

        # prepare callbacks
        monitor_value_name = ("val_loss" if val_generator is not None else "loss")
        checkpoint_filepath = os.path.join(run_output_dir, "best_checkpoint.hdf5")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor=monitor_value_name,
            verbose=1,
            save_best_only=True)
        visualization_filepath = os.path.join(run_output_dir, "visualizations supervised")
        os.mkdir(visualization_filepath)
        visualization_callbacks = [
            VisualizationCallback(
                test_batches=train_generator,
                delta_steps=0,
                output_directory=visualization_filepath,
                epochs_per_visualization=1,
                visualize_first_n=visualize_first_n),
            VisualizationCallback(
                test_batches=train_generator,
                delta_steps=delta_steps,
                output_directory=visualization_filepath,
                epochs_per_visualization=1,
                visualize_first_n=visualize_first_n)]
        csv_logging_filepath = os.path.join(run_output_dir, "epoch_results_supervised.csv")
        csv_logging_callback = tf.keras.callbacks.CSVLogger(csv_logging_filepath)
        terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_value_name,
            min_delta=early_stop_min_delta,
            patience=early_stop_patience,
            restore_best_weights=True)
        callbacks = [
            checkpoint_callback,
            *visualization_callbacks,
            csv_logging_callback,
            terminate_on_nan_callback,
            early_stopping_callback]

        # train the model
        '''
        if val_generator is not None:
            reverse_model.fit(
                x=train_generator,
                epochs=max_epochs,
                callbacks=callbacks,
                validation_data=val_generator,
                validation_freq=val_freq)
        else:
            reverse_model.fit(
                x=train_generator,
                epochs=max_epochs,
                callbacks=callbacks)
        '''

        # recompile the model with a new loss
        unsupervised_loss_fn = TrueTargetLossFn(delta_steps=1, name=None, y_true_is_start=True)
        reverse_model.compile(optimizer="adam", loss=unsupervised_loss_fn, metrics=acc_fns)
        reverse_model.build(input_shape=(None, 25, 25, 1))
        reverse_model.summary()

        # make a new visualization directory, recreate callbacks
        visualization_filepath = os.path.join(run_output_dir, "visualizations unsupervised")
        os.mkdir(visualization_filepath)
        visualization_callbacks = [
            VisualizationCallback(
                test_batches=train_generator,
                delta_steps=0,
                output_directory=visualization_filepath,
                epochs_per_visualization=1,
                visualize_first_n=visualize_first_n),
            VisualizationCallback(
                test_batches=train_generator,
                delta_steps=delta_steps,
                output_directory=visualization_filepath,
                epochs_per_visualization=1,
                visualize_first_n=visualize_first_n)]
        csv_logging_filepath = os.path.join(run_output_dir, "epoch_results_unsupervised.csv")
        csv_logging_callback = tf.keras.callbacks.CSVLogger(csv_logging_filepath)
        callbacks = [
            checkpoint_callback,
            *visualization_callbacks,
            csv_logging_callback,
            terminate_on_nan_callback,
            early_stopping_callback]

        # train the model again
        if val_generator is not None:
            reverse_model.fit(
                x=train_generator,
                epochs=max_epochs,
                callbacks=callbacks,
                validation_data=val_generator,
                validation_freq=val_freq)
        else:
            reverse_model.fit(
                x=train_generator,
                epochs=max_epochs,
                callbacks=callbacks)

        # save the trained model
        saved_model_directory = os.path.join(run_output_dir, "saved_model")
        os.mkdir(saved_model_directory)
        reverse_model.save(saved_model_directory)

        # finalize folder name
        os.rename(run_output_dir, run_output_dir_complete)


if __name__ == "__main__":
    TrainingRunnerTransition.run(
        root_output_dir=r"D:\Documents\Reverse Conway\Output",
        delta_steps=1,
        train_generator_name="KaggleSupervisedDeltaOneDataGenerator",
        train_generator_config={"batch_size": 128, "samples_per_epoch": 2**16},
        model_name="BaselineConvModel",
        model_config={"n_filters": 128, "n_layers": 24},
        early_stop_patience=20,
        early_stop_min_delta=0.001,
        max_epochs=1000,
        visualize_first_n=5)




    # TODO: TRY 37 layers, ideal!



