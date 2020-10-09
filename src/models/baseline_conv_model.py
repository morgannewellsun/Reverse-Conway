import tensorflow as tf

from components.roll_padding_2d_layer import RollPadding2DLayer


class BaselineConvModel(tf.keras.Sequential):

    def __init__(self, n_filters: int, n_layers: int):
        super(BaselineConvModel, self).__init__()
        self._n_filters = n_filters
        self._n_hidden_layers = n_layers - 1
        self.add(RollPadding2DLayer(1))
        for _ in range(n_layers - 1):
            self.add(tf.keras.layers.Conv2D(n_filters, (3, 3), activation="relu"))
            self.add(RollPadding2DLayer(1))
            self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid"))

    def get_config(self):
        return {"n_filters": self._n_filters, "n_layers": self._n_hidden_layers + 1}


if __name__ == "__main__":
    pass
