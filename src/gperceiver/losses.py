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


class PloidyBinaryCrossentropy(tf.keras.losses.BinaryCrossentropy):

    def __init__(
        self,
        ploidy=2,
        ploidy_scaler=1.0,
        from_logits=False,
        label_smoothing=0.,
        axis=-1,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="ploidy_binary_crossentropy",
    ):

        super().__init__(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
            name=name,
        )
        self.axis = axis
        self.ploidy = ploidy
        self.ploidy_scaler = ploidy_scaler
        self.from_logits = from_logits
        return

    def call(self, ytrue, ypred):
        loss = super().call(ytrue, ypred)

        if self.from_logits:
            ypred = tf.sigmoid(ypred)

        loss += (
            self.ploidy_scaler *
            tf.math.squared_difference(
                tf.convert_to_tensor(self.ploidy, dtype=ypred.dtype),
                tf.reduce_sum(ypred, axis=self.axis)
            )
        )
        return loss

    def get_config(self):
        config = super().get_config()
        config["ploidy"] = self.ploidy
        config["ploidy_scaler"] = self.ploidy_scaler
        return config


class PloidyBinaryFocalCrossentropy(tf.keras.losses.BinaryFocalCrossentropy):

    def __init__(
        self,
        ploidy=2,
        ploidy_scaler=1.0,
        gamma: float = 2.0,
        from_logits=False,
        label_smoothing=0.,
        axis=-1,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="ploidy_binary_focal_crossentropy",
    ):

        super().__init__(
            gamma=gamma,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
            name=name,
        )
        self.axis = axis
        self.ploidy = ploidy
        self.ploidy_scaler = ploidy_scaler
        self.from_logits = from_logits
        return

    def call(self, ytrue, ypred):
        loss = super().call(ytrue, ypred)

        if self.from_logits:
            ypred = tf.sigmoid(ypred)

        loss += (
            self.ploidy_scaler *
            tf.math.squared_difference(
                tf.convert_to_tensor(self.ploidy, dtype=ypred.dtype),
                tf.reduce_sum(ypred, axis=self.axis)
            )
        )
        return loss

    def get_config(self):
        config = super().get_config()
        config["ploidy"] = self.ploidy
        config["ploidy_scaler"] = self.ploidy_scaler
        return config
