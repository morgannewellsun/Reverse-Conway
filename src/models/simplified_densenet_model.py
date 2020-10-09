import tensorflow as tf

from components.roll_padding_2d_layer import RollPadding2DLayer


class DenseBlock(tf.keras.Model):

    def __init__(self, n_filters: int, is_input: bool = False, is_output: bool = False):
        super(DenseBlock, self).__init__()
        self._n_filters = n_filters
        self._is_input = is_input
        self._is_output = is_output
        self._batch_norm_layer = tf.keras.layers.BatchNormalization() if not is_input else None
        self._roll_padding_layer = RollPadding2DLayer(padding=1)
        self._convolution_layer = (
                (tf.keras.layers.Conv2D(n_filters, (3, 3), activation="relu"))
                if not is_output
                else (tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid")))
        self._concatenate_layer = tf.keras.layers.Concatenate(axis=-1) if not is_output else None

    def call(self, inputs, **kwargs):
        original_inputs = inputs
        if not self._is_input:
            inputs = self._batch_norm_layer(inputs)
        inputs = self._roll_padding_layer(inputs)
        inputs = self._convolution_layer(inputs)
        if self._is_output:
            inputs = self._concatenate_layer([original_inputs, inputs])
        return inputs

    def get_config(self):
        return {"n_filters": self._n_filters, "is_input": self._is_input, "is_output": self._is_output}


class SimplifiedDenseNetModel(tf.keras.Sequential):

    def __init__(self, growth_rate: int, n_layers: int):
        super(SimplifiedDenseNetModel, self).__init__()
        self._growth_rate = growth_rate
        self._n_layers = n_layers
        for i in range(n_layers):
            self.add(DenseBlock(growth_rate, is_input=(i == 0), is_output=(i == n_layers - 1)))

    def get_config(self):
        return {"growth_rate": self._growth_rate, "n_layers": self._n_layers}


if __name__ == "__main__":
    pass
