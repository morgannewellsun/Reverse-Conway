import datetime as dt
import json
import os

import tensorflow as tf

from components.true_target_acc_fn import TrueTargetAccFn
from components.true_target_loss_fn import TrueTargetLossFn
from data.kaggle_data_generator import KaggleDataGenerator
from data.pretty_test_target import pretty_test_target
from framework.visualization_callback import VisualizationCallback
from framework.visualizer import Visualizer
from models.baseline_conv_model import BaselineConvModel


class TrainingRunner:

    @staticmethod
    def run(
            root_output_dir: str,
            delta_steps: int,
            generator_name: str,
            generator_config: dict,
            model_name: str,
            model_config: dict,
            early_stop_patience: int,
            max_epochs: int
    ) -> None:

        # cache run config parameters
        run_config_dict = locals().copy()

        # create run output directory
        run_folder_name = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        run_folder_name = os.path.join(run_folder_name, "")
        run_output_dir = os.path.join(root_output_dir, run_folder_name)
        os.mkdir(run_output_dir)

        # save run config parameters
        run_config_filepath = os.path.join(run_output_dir, "run_config.json")
        with open(run_config_filepath, "w") as file:
            json.dump(run_config_dict, file, indent=4)

        # prepare data generator
        if generator_name == "KaggleDataGenerator":
            train_generator = KaggleDataGenerator(
                delta_steps=delta_steps, **generator_config)
        else:
            raise ValueError(f"{generator_name} is not a supported data generator.")

        # prepare the model
        if model_name == "BaselineConvModel":
            reverse_model = BaselineConvModel(**model_config)
        else:
            raise ValueError(f"{model_name} is not a supported model.")
        loss_fn = TrueTargetLossFn(delta_steps=delta_steps, name="TrueTargetLoss")
        acc_fn = TrueTargetAccFn(delta_steps=delta_steps, name="TrueTargetAcc")
        reverse_model.compile(optimizer="adam", loss=loss_fn, metrics=[acc_fn])

        # prepare callbacks
        checkpoint_filepath = os.path.join(run_output_dir, "best_checkpoint.hdf5")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="TrueTargetAcc",
            verbose=1,
            save_best_only=True)
        visualization_filepath = os.path.join(run_output_dir, "visualizations")
        os.mkdir(visualization_filepath)
        visualization_callback = VisualizationCallback(
            test_batches=train_generator,
            delta_steps=delta_steps,
            output_directory=visualization_filepath,
            epochs_per_visualization=1)
        pretty_test_target_callback = VisualizationCallback(
            test_batches=None,
            delta_steps=delta_steps,
            output_directory=visualization_filepath,
            epochs_per_visualization=5,
            use_pretty_test_target=True)
        csv_logging_filepath = os.path.join(run_output_dir, "epoch_results.csv")
        csv_logging_callback = tf.keras.callbacks.CSVLogger(csv_logging_filepath)
        terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="TrueTargetAcc",
            patience=early_stop_patience,
            restore_best_weights=True)
        callbacks = [
            checkpoint_callback,
            visualization_callback,
            pretty_test_target_callback,
            csv_logging_callback,
            terminate_on_nan_callback,
            early_stopping_callback]

        # train the model
        reverse_model.fit(x=train_generator, epochs=max_epochs, callbacks=callbacks)

        # save the trained model
        saved_model_directory = os.path.join(run_output_dir, "saved_model")
        os.mkdir(saved_model_directory)
        reverse_model.save(saved_model_directory)

        # ensure saved model can be loaded and used
        reverse_model = tf.keras.models.load_model(saved_model_directory, compile=False)
        test_prediction = reverse_model(pretty_test_target).numpy().squeeze()
        # test_visualizer = Visualizer(show_figures=True, save_directory=None)
        # test_visualizer.draw_board(board=test_prediction, title="test_prediction")


if __name__ == "__main__":
    for delta_steps in [1, 2, 3, 4, 5]:
        TrainingRunner.run(
            root_output_dir=r"D:\Documents\Reverse Conway\Output",
            delta_steps=delta_steps,
            generator_name="KaggleDataGenerator",
            generator_config={"batch_size": 256, "verbose": True},
            model_name="BaselineConvModel",
            model_config={"n_filters": 128, "n_hidden_layers": 10},
            early_stop_patience=20,
            max_epochs=1000)




