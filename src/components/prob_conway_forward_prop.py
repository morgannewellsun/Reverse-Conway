import itertools
import tensorflow as tf


class ProbConwayForwardProp(tf.keras.layers.Layer):

    def __init__(self):
        super(ProbConwayForwardProp, self).__init__()
        self.moore_offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i != 0 or j != 0)]
        self.indices_and_complements_8c2 = self._generate_indices_and_complements_8cr(2)
        self.indices_and_complements_8c3 = self._generate_indices_and_complements_8cr(3)

    def call(self, inputs, **kwargs):
        probs_live_neighbors = [tf.roll(inputs, shift, (-2, -1)) for shift in self.moore_offsets]
        probs_dead_neighbors = [tf.math.subtract(1, probs) for probs in probs_live_neighbors]
        probs_exactly_two_live_neighbors = tf.reduce_sum(
            axis=0,
            input_tensor=[
                tf.reduce_prod(
                    axis=0,
                    input_tensor=(
                            [probs_live_neighbors[i] for i in indices]
                            + [probs_dead_neighbors[c] for c in complements]))
                for indices, complements in zip(*self.indices_and_complements_8c2)])
        probs_exactly_three_live_neighbors = tf.reduce_sum(
            axis=0,
            input_tensor=[
                tf.reduce_prod(
                    axis=0,
                    input_tensor=(
                            [probs_live_neighbors[i] for i in indices]
                            + [probs_dead_neighbors[c] for c in complements]))
                for indices, complements in zip(*self.indices_and_complements_8c3)])
        probs_live_self_next_step = tf.reduce_sum(
            axis=0,
            input_tensor=[
                probs_exactly_three_live_neighbors,
                tf.reduce_prod(
                    [probs_exactly_two_live_neighbors, inputs], axis=0)])
        return probs_live_self_next_step

    @staticmethod
    def _generate_indices_and_complements_8cr(r: int):
        indices_8cr = [list(i) for i in itertools.combinations(range(8), r)]
        complements_8cr = [list([i for i in range(8) if i not in ixs]) for ixs in indices_8cr]
        return indices_8cr, complements_8cr
