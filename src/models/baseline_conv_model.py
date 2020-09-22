import tensorflow as tf

from components.prob_conway_forward_prop import ProbConwayForwardProp
from components.roll_padding_2d import RollPadding2D


class BaselineConvModel(tf.keras.Model):

    def __init__(self, n_filters: int, n_hidden_layers: int):
        super(BaselineConvModel, self).__init__()
        self._layers = []
        for _ in range(n_hidden_layers):
            self._layers.append(RollPadding2D(1))
            self._layers.append(
                tf.keras.layers.Conv2D(n_filters, (3, 3), activation="relu"))
            self._layers.append(tf.keras.layers.BatchNormalization())
        self._layers.append(RollPadding2D(1))
        self._layers.append(tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid"))
        self._wrapped_model = tf.keras.Sequential(self._layers)

    def call(self, inputs, **kwargs):
        return self._wrapped_model(inputs)


if __name__ == "__main__":
    import numpy as np
    np_dummy_data = np.random.random(size=(100, 25, 25, 1)).astype(np.float32)
    tf_dummy_data = tf.constant(np_dummy_data)
    model = BaselineConvModel(n_filters=3, n_hidden_layers=3)
    dummy_output = model(tf_dummy_data)
    np.set_printoptions(precision=3, linewidth=1000)
    print(dummy_output.numpy()[0, :, :, 0])
    print(dummy_output.shape)