import tensorflow as tf

from components.roll_padding_2d_layer import RollPadding2DLayer


class _ResNeXtBlock(tf.keras.Model):

    def __init__(self, dimension: int, cardinality: int):
        super(_ResNeXtBlock, self).__init__()

        self._dimension = dimension
        self._cardinality = cardinality

        self._first_local_dense_layer = tf.keras.layers.Conv2D(4 * cardinality, (1, 1))
        self._first_local_dense_bn = tf.keras.layers.BatchNormalization()
        self._first_local_dense_af = tf.keras.layers.ReLU()

        self._grouped_convolution_pad = RollPadding2DLayer(padding=1)
        self._grouped_convolution_layer = tf.keras.layers.Conv2D(4 * cardinality, (3, 3), groups=cardinality)
        self._grouped_convolution_bn = tf.keras.layers.BatchNormalization()
        self._grouped_convolution_af = tf.keras.layers.ReLU()

        self._second_local_dense_layer = tf.keras.layers.Conv2D(dimension, (1, 1))
        self._second_local_dense_bn = tf.keras.layers.BatchNormalization()

        self._output_sum = tf.keras.layers.Add()
        self._output_af = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):

        original_inputs = inputs

        inputs = self._first_local_dense_layer(inputs)
        inputs = self._first_local_dense_bn(inputs)
        inputs = self._first_local_dense_af(inputs)

        inputs = self._grouped_convolution_pad(inputs)
        inputs = self._grouped_convolution_layer(inputs)
        inputs = self._grouped_convolution_bn(inputs)
        inputs = self._grouped_convolution_af(inputs)

        inputs = self._second_local_dense_layer(inputs)
        inputs = self._second_local_dense_bn(inputs)

        inputs = self._output_sum([original_inputs, inputs])
        inputs = self._output_af(inputs)

        return inputs

    def get_config(self):
        return {"dimension": self._dimension, "cardinality": self._cardinality}


class SimplifiedResNeXtModel(tf.keras.Sequential):

    def __init__(self, dimension: int, cardinality: int, n_blocks: int):
        super(SimplifiedResNeXtModel, self).__init__()
        self._dimension = dimension
        self._cardinality = cardinality
        self._n_blocks = n_blocks
        for _ in range(n_blocks):
            self.add(_ResNeXtBlock(dimension, cardinality))
        self.add(tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid"))

    def get_config(self):
        return {"dimension": self._dimension, "cardinality": self._cardinality, "n_blocks": self._n_blocks}


if __name__ == "__main__":
    pass
