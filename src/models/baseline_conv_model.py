import tensorflow as tf

from components.roll_padding_2d_layer import RollPadding2DLayer


class BaselineConvModel(tf.keras.Model):

    def __init__(self, n_filters: int, n_hidden_layers: int):
        super(BaselineConvModel, self).__init__()
        self._n_filters = n_filters
        self._n_hidden_layers = n_hidden_layers
        self._wrapped_model = tf.keras.Sequential()
        for _ in range(n_hidden_layers):
            self._wrapped_model.add(RollPadding2DLayer(1))
            self._wrapped_model.add(
                tf.keras.layers.Conv2D(n_filters, (3, 3), activation="relu"))
            self._wrapped_model.add(tf.keras.layers.BatchNormalization())
        self._wrapped_model.add(RollPadding2DLayer(1))
        self._wrapped_model.add(tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid"))

    def call(self, inputs, **kwargs):
        return self._wrapped_model(inputs)

    def get_config(self):
        return {"n_filters": self._n_filters, "n_hidden_layers": self._n_hidden_layers}


if __name__ == "__main__":
    import numpy as np
    np_dummy_data = np.random.random(size=(100, 25, 25, 1)).astype(np.float32)
    tf_dummy_data = tf.constant(np_dummy_data)
    model = BaselineConvModel(n_filters=3, n_hidden_layers=3)
    dummy_output = model(tf_dummy_data)
    np.set_printoptions(precision=3, linewidth=1000)
    print(dummy_output.numpy()[0, :, :, 0])
    print(dummy_output.shape)
