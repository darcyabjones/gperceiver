import numpy as np
import tensorflow as tf


def pairwise_correlation(a, offset=1, metric="hamming"):
    if metric == "hamming":
        from scipy.spatial.distance import hamming

        def m(x, y):
            return min([hamming(x, y), hamming(2-x, y)])

    elif metric == "cor":

        def m(x, y):
            c = np.corrcoef(x, y)[0, 1]
            if np.isnan(c):
                c = 0.0
            return c
    else:
        raise ValueError("hamming or cor")

    out = np.zeros(a.shape[1])
    for i in range(0, a.shape[1] - offset):
        j = i + offset

        z = m(a[:, i], a[:, j])
        out[j] = z
    return out.tolist()


class PrepAECOneHot(object):

    def __init__(
        self,
        input_dim: int = None,
        offset: int = 1,
        prop_ones: float = 0.5,
        onehot: bool = True,
        dtype=tf.float32
    ):
        if onehot and (input_dim is None):
            raise ValueError("If onehot encoding, input_dim must be specified")
        self.offset = offset
        self.input_dim = input_dim
        self.prop_ones = prop_ones
        self.onehot = onehot
        self.dtype = dtype
        return

    @tf.function
    def __call__(self, x, w=None):
        """Prepares an auto-encoding type task"""

        x_dropped = x + self.offset

        # Generates a sample of 1 and 0. Most will be 1.
        samples = tf.cast(
            tf.random.categorical(
                tf.math.log([[1. - self.prop_ones, self.prop_ones]]),
                tf.shape(x_dropped)[0]
            ),
            dtype=x_dropped.dtype
        )[0]

        x_dropped = samples * x_dropped

        if self.onehot:
            x_dropped = tf.one_hot(
                x_dropped,
                self.input_dim + self.offset,
                dtype=self.dtype
            )

        if w is None:
            return x_dropped, x
        else:
            return x_dropped, x, w


class PrepAEC(object):

    def __init__(
        self,
        offset: int = 1,
        prop_ones: float = 0.8,
        dtype=tf.float32,
        seed=None,
    ):
        self.offset = offset
        self.prop_ones = prop_ones
        self.dtype = dtype
        if seed is None:
            import random
            self.seed = random.getstate()[1][0]
        else:
            self.seed = seed
        return

    def __call__(self, x, w=None):
        """Prepares an auto-encoding type task"""

        seq_len = tf.shape(x)[-1]
        num_to_sample = tf.cast(
            tf.math.round(
                (1.0 - self.prop_ones) *
                tf.cast(seq_len, tf.float32)
            ),
            tf.int32
        )
        x_dropped = x + self.offset

        # Generates a sample of 1 and 0. Most will be 1.
        samples = tf.cast(
            tf.random.categorical(
                tf.math.log([[1. - self.prop_ones, self.prop_ones]]),
                seq_len,
                seed=self.seed
            ),
            dtype=x_dropped.dtype
        )[0]

        indices = tf.range(0, seq_len, dtype=tf.int32)
        test_indices = tf.cast(tf.where(
            tf.logical_not(tf.cast(samples, tf.bool))
        )[:, 0], tf.int32)

        if tf.size(test_indices, tf.int32) < num_to_sample:
            extras = tf.random.stateless_uniform(
                ((num_to_sample - tf.size(test_indices, tf.int32)),),
                tf.random.get_global_generator().state[:2],
                minval=0,
                maxval=seq_len,
                dtype=tf.dtypes.int32
            )
            test_indices = tf.concat([test_indices, extras], 0)
        elif tf.size(test_indices, tf.int32) > num_to_sample:
            extras = tf.random.stateless_uniform(
                (num_to_sample,),
                tf.random.get_global_generator().state[:2],
                minval=0,
                maxval=tf.size(test_indices),
                dtype=tf.dtypes.int32
            )
            test_indices = tf.cast(tf.gather(test_indices, extras), tf.int32)

        y = tf.gather(x, test_indices, axis=-1)
        x_dropped = samples * x_dropped

        if w is None:
            return (x_dropped, indices, test_indices), y
        else:
            return (x_dropped, indices, test_indices), y, w


@tf.function
def prep_y(x, y, w=None):
    x, g = x
    x = x + 1
    x = tf.one_hot(x, 4, dtype=tf.dtypes.float32)

    if w is None:
        return (x, g), y
    else:
        return (x, g), y, w


@tf.function
def prep_selfsupervised_aec(x, y, w=None):
    x, g = x
    x_dropped = x + 1

    # Generates a sample of 1 and 0. Most will be 1.
    samples = tf.cast(
        tf.random.categorical(
            tf.math.log([[0.2, 0.8]]),
            tf.shape(x_dropped)[0]
        ),
        dtype=x_dropped.dtype
    )

    x_dropped = samples * x_dropped
    x_dropped = tf.one_hot(x_dropped, 4, dtype=tf.dtypes.float32)
    y2 = tf.one_hot(x, 3, dtype=tf.dtypes.float32)

    if w is None:
        return (x_dropped, g), (y, y2)
    else:
        return (x_dropped, g), (y, y2), w
