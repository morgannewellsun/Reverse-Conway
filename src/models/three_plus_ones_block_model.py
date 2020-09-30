import tensorflow as tf

from components.roll_padding_2d_layer import RollPadding2DLayer


class ThreePlusOnesBlockModel(tf.keras.Sequential):

    def __init__(self, n_filters: int, n_blocks: int, n_layers_per_block: int):
        super(ThreePlusOnesBlockModel, self).__init__()
        self._n_blocks = n_blocks
        self._n_filters = n_filters
        self._n_layers_per_block = n_layers_per_block
        self.add(RollPadding2DLayer(1))
        self.add(tf.keras.layers.Conv2D(n_filters, (3, 3), activation="relu"))
        for i in range(n_blocks):
            for _ in range(n_layers_per_block - 1):
                self.add(tf.keras.layers.BatchNormalization())
                self.add(tf.keras.layers.Conv2D(n_filters, (1, 1), activation="relu"))
            if i < n_blocks - 1:
                self.add(RollPadding2DLayer(1))
                self.add(tf.keras.layers.BatchNormalization())
                self.add(tf.keras.layers.Conv2D(n_filters, (3, 3), activation="relu"))
        self.add(tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid"))

    def get_config(self):
        return {
            "n_filters": self._n_filters,
            "n_blocks": self._n_blocks,
            "n_layers_per_block": self._n_layers_per_block}


if __name__ == "__main__":
    pass


"""
regarding n_blocks = delta_steps:

I initially expected this to perform better than the baseline conv
model, because the model architecture seems to align with the dynamics
of Conway's game of life. However, this performed much worse than the
baseline model. This teaches us something important: Even though 
Conway can be deterministically solved in the forward direction using
3x3 convolutions, the reverse direction requires context from the whole
board, not just the 3x3 neighborhood.
"""
