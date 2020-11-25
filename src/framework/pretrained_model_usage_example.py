import tensorflow as tf

from data.pretty_test_target import pretty_test_target
from framework.visualizer import Visualizer

# load model
path_to_saved_model = r'D:\Documents\Reverse Conway\Reverse-Conway\pretrained_models\crossfade_baseline_delta_1'
# loaded_model = tf.keras.models.load_model(path_to_saved_model, compile=False)  # compile=True will fail!
loaded_model = tf.saved_model.load(path_to_saved_model)

# inspect test input
print(pretty_test_target.shape)

# perform prediction
test_prediction = loaded_model(pretty_test_target).numpy()

# inspect output
print(test_prediction.shape)
vis = Visualizer(show_figures=True, save_directory=None)
vis.draw_board(test_prediction.squeeze(), "test")
