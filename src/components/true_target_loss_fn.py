import tensorflow as tf

from components.prob_conway_forward_prop_fn import ProbConwayForwardPropFn


class TrueTargetLossFn(tf.keras.losses.Loss):

    def __init__(self, delta_steps: int, name: str):
        super(TrueTargetLossFn, self).__init__(name=name)
        self._delta_steps = delta_steps
        self._prob_forward_prop = ProbConwayForwardPropFn()
        self._bce_loss_fn = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        for _ in range(self._delta_steps):
            y_pred = self._prob_forward_prop(y_pred)
        true_target_loss = self._bce_loss_fn(y_true, y_pred)
        return true_target_loss

    def get_config(self):
        return {"delta_steps": self._delta_steps, "name": self.name}
