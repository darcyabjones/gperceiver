import tensorflow as tf


class ContrastiveLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.AUTO,
        name=None,
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        return

    def prep_tiled(self, y):

        tiled = tf.tile(y, (tf.shape(y)[0], 1), name="tile")
        repeated = tf.repeat(y, tf.shape(y)[0], axis=0, name="repeat")
        diffs = repeated - tiled
        return diffs

    def call(self, y_true, y_pred):
        if len(tf.shape(y_pred)) == 3:
            # assert int(tf.shape(y_pred)[-1]) == 1

            y_pred = tf.reshape(
                y_pred,
                (tf.shape(y_pred)[0], tf.shape(y_pred)[1])
            )

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_tiled = self.prep_tiled(y_true)
        y_pred_tiled = self.prep_tiled(y_pred)

        diffs = tf.math.squared_difference(y_true_tiled, y_pred_tiled)
        diffs = tf.reduce_sum(diffs, axis=-1)
        batch_size = tf.shape(y_pred)[0]
        diffs = tf.reshape(diffs, (batch_size, batch_size))
        diffs = tf.reduce_mean(diffs, axis=-1)
        return diffs
