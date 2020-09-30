import tensorflow as tf

from data.pretty_test_target import pretty_test_target


if __name__ == "__main__":

    # load model
    path_to_saved_model = r"D:\Documents\Reverse Conway\Reverse-Conway\pretrained_models\initial_baseline_delta_1"
    loaded_model = tf.keras.models.load_model(path_to_saved_model, compile=False)  # compile=True will fail!

    # inspect test input
    print(pretty_test_target.shape)

    # perform prediction
    test_prediction = loaded_model(pretty_test_target).numpy()

    # inspect output
    print(test_prediction.shape)
