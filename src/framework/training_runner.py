import datetime as dt
import json
import os

import tensorflow as tf

from components.true_target_acc_fn import TrueTargetAccFn
from components.true_target_loss_fn import TrueTargetLossFn
from data.pretty_test_target import pretty_test_target
from data.kaggle_supervised_data_generator import KaggleSupervisedDataGenerator
from data.kaggle_unsupervised_data_generator import KaggleUnsupervisedDataGenerator
from framework.visualization_callback import VisualizationCallback
from framework.visualizer import Visualizer
from models.baseline_conv_model import BaselineConvModel
from models.three_plus_ones_block_model import ThreePlusOnesBlockModel


class TrainingRunner:

    @staticmethod
    def run(
            root_output_dir: str,
            delta_steps: int,
            supervision: bool,
            generator_config: dict,
            model_name: str,
            model_config: dict,
            early_stop_patience: int,
            early_stop_min_delta: float,
            max_epochs: int
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

        # prepare data generator
        if supervision:
            train_generator = KaggleSupervisedDataGenerator(delta_steps=delta_steps, **generator_config)
        else:
            train_generator = KaggleUnsupervisedDataGenerator(delta_steps=delta_steps, **generator_config)

        # prepare the model
        if model_name == "BaselineConvModel":
            reverse_model = BaselineConvModel(**model_config)
        elif model_name == "ThreePlusOnesBlockModel":
            reverse_model = ThreePlusOnesBlockModel(**model_config)
        else:
            raise ValueError(f"{model_name} is not a supported model.")
        if supervision:
            loss_fn = tf.keras.losses.BinaryCrossentropy()  # bce loss monitored
            acc_fn = TrueTargetAccFn(delta_steps=0, name="StartAcc")
            reverse_model.compile(optimizer="adam", loss=loss_fn, metrics=[acc_fn])
        else:
            loss_fn = TrueTargetLossFn(delta_steps=delta_steps, name="TrueTargetLoss")
            acc_fn = TrueTargetAccFn(delta_steps=delta_steps, name="TrueTargetAcc")  # stop tt acc monitored
            reverse_model.compile(optimizer="adam", loss=loss_fn, metrics=[acc_fn])
        reverse_model.build(input_shape=(None, 25, 25, 1))
        reverse_model.summary()

        # prepare callbacks
        monitor_value_name = "loss" if supervision else "TrueTargetAcc"
        checkpoint_filepath = os.path.join(run_output_dir, "best_checkpoint.hdf5")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor=monitor_value_name,
            verbose=1,
            save_best_only=True)
        visualization_filepath = os.path.join(run_output_dir, "visualizations")
        os.mkdir(visualization_filepath)
        visualization_callback = VisualizationCallback(
            test_batches=train_generator,
            delta_steps=(0 if supervision else delta_steps),
            output_directory=visualization_filepath,
            epochs_per_visualization=1)
        csv_logging_filepath = os.path.join(run_output_dir, "epoch_results.csv")
        csv_logging_callback = tf.keras.callbacks.CSVLogger(csv_logging_filepath)
        terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_value_name,
            min_delta=early_stop_min_delta,
            patience=early_stop_patience,
            restore_best_weights=True)
        callbacks = [
            checkpoint_callback,
            visualization_callback,
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

        # finalize folder name
        os.rename(run_output_dir, run_output_dir_complete)


if __name__ == "__main__":
    import time
    for supervision in [True]:
        for delta_steps in [1, 2, 3, 4, 5]:
            for n_filters in [256]:
                for n_layers in [12]:
                    time.sleep(1)
                    try:
                        TrainingRunner.run(
                            root_output_dir=r"D:\Documents\Reverse Conway\Output",
                            delta_steps=delta_steps,
                            supervision=supervision,
                            generator_config={
                                "batch_size": 256,
                                "samples_per_epoch": 2**16,
                                "verbose": True},
                            model_name="BaselineConvModel",
                            model_config={"n_filters": n_filters, "n_layers": n_layers},
                            early_stop_patience=20,
                            early_stop_min_delta=0.0001,
                            max_epochs=1000)
                    except Exception as e:
                        for _ in range(64):
                            print("SOMETHING WENT WRONG!")
                        print(e)

    # BaselineConvModel: {"n_filters": 1, "n_layers": 1}
    # ThreePlusOnesBlockModel: {"n_filters": 1, "n_blocks": 1, "n_layers_per_block": 1}





