import tensorflow as tf


class RollPadding2DLayer(tf.keras.layers.Layer):

    def __init__(self, padding: int):
        super(RollPadding2DLayer, self).__init__()
        self._padding = padding
        self._left_begin_arg = None
        self._left_size_arg = None
        self._right_begin_arg = None
        self._right_size_arg = None
        self._top_begin_arg = None
        self._top_size_arg = None
        self._bottom_begin_arg = None
        self._bottom_size_arg = None

    def build(self, input_shape):
        if len(input_shape) == 3:
            batch_begin_arg, batch_size_arg = [], []
        elif len(input_shape) == 4:
            batch_begin_arg, batch_size_arg = [0], [-1]
        else:
            raise ValueError(f"RollPadding2DLayer recieved an input of shape {input_shape}. Dimension must be 3 or 4.")
        # x dimension
        self._left_begin_arg = tf.constant(batch_begin_arg + [0, 0, 0])
        self._left_size_arg = tf.constant(batch_size_arg + [input_shape[-3], self._padding, input_shape[-1]])
        self._right_begin_arg = tf.constant(batch_begin_arg + [0, input_shape[-2] - self._padding, 0])
        self._right_size_arg = tf.constant(batch_size_arg + [input_shape[-3], self._padding, input_shape[-1]])
        # y dimension
        self._top_begin_arg = tf.constant(batch_begin_arg + [0, 0, 0])
        self._top_size_arg = tf.constant(batch_size_arg + [self._padding, input_shape[-2] + 2 * self._padding, input_shape[-1]])
        self._bottom_begin_arg = tf.constant(batch_begin_arg + [input_shape[-3] - self._padding, 0, 0])
        self._bottom_size_arg = tf.constant(batch_size_arg + [self._padding, input_shape[-2] + 2 * self._padding, input_shape[-1]])

    def call(self, inputs, **kwargs):
        left_slice = tf.slice(inputs, self._left_begin_arg, self._left_size_arg)
        right_slice = tf.slice(inputs, self._right_begin_arg, self._right_size_arg)
        x_dim_padded_inputs = tf.concat([right_slice, inputs, left_slice], axis=-2)
        top_slice = tf.slice(x_dim_padded_inputs, self._top_begin_arg, self._top_size_arg)
        bottom_slice = tf.slice(x_dim_padded_inputs, self._bottom_begin_arg, self._bottom_size_arg)
        xy_dim_padded_inputs = tf.concat([bottom_slice, x_dim_padded_inputs, top_slice], axis=-3)
        return xy_dim_padded_inputs

    def get_config(self):
        return {"padding": self._padding}
