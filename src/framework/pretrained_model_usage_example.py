import tensorflow as tf

from data.pretty_test_target import pretty_test_target
from framework.visualizer import Visualizer

# load model
path_to_saved_model = r'D:\Documents\Reverse Conway\Output\15. largest densenet with growth rate of 8, promising results\20201013T100114 (delta 1)\saved_model'
loaded_model = tf.keras.models.load_model(path_to_saved_model, compile=False)  # compile=True will fail!

# inspect test input
print(pretty_test_target.shape)

# perform prediction
test_prediction = loaded_model(pretty_test_target).numpy()

# inspect output
print(test_prediction.shape)
vis = Visualizer(show_figures=True, save_directory=None)
vis.draw_board(test_prediction.squeeze(), "test")
