import tensorflow as tf

from components.prob_conway_forward_prop_fn import ProbConwayForwardPropFn


class TrueTargetLossFn:

    __name__ = "TrueTargetLoss"

    def __init__(self, delta_steps: int):
        self._delta_steps = delta_steps
        self._prob_forward_prop = ProbConwayForwardPropFn()
        self._bce_loss_fn = tf.keras.losses.BinaryCrossentropy()

    def __call__(self, y_true, y_pred):
        for _ in range(self._delta_steps):
            y_pred = self._prob_forward_prop(y_pred)
        true_target_loss = self._bce_loss_fn(y_true, y_pred)
        return true_target_loss
