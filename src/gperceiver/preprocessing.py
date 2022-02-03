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


class DecodeAlleles(object):

    def __init__(self, int_decoder: "tf.Tensor", dtype=tf.dtypes.uint8):
        self.int_decoder = tf.convert_to_tensor(int_decoder, dtype=dtype)
        return

    def __call__(self, X):
        return tf.gather(self.int_decoder, X)


class AlleleCounts(object):

    def __init__(self):
        return

    def __call__(self, X):
        targets = tf.reduce_sum(tf.one_hot(X - 1, 2), axis=-2)
        count_targets = tf.cast(
            (tf.expand_dims(targets, -1) - tf.range(2, dtype=tf.float32)) > 0,
            tf.float32
        )

        shape = tf.shape(X)
        return tf.reshape(count_targets, (shape[0], shape[1], 4))




class PrepAEC(object):

    def __init__(
        self,
        offset: int = 1,
        prop_ones: float = 0.8,
        dtype=tf.float32,
        seed=None,
    ):
        import random
        self.offset = offset
        self.prop_ones = prop_ones
        self.dtype = dtype
        self.rng = random.Random(seed)
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
                seed=self.rng.getrandbits(128)
            ),
            dtype=x_dropped.dtype
        )[0]

        indices = tf.range(0, seq_len, dtype=tf.int32)

        # Only include >=0 to avoid real missing data as targets
        test_indices = tf.cast(tf.where(
            tf.logical_and(tf.logical_not(tf.cast(samples, tf.bool)), x >= 0)
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

        if len(tf.shape(x)) == 2:
            indices = tf.repeat(
                tf.expand_dims(indices, 0),
                tf.shape(x)[0],
                axis=0
            )
            test_indices = tf.repeat(
                tf.expand_dims(test_indices, 0),
                tf.shape(x)[0],
                axis=0
            )

        if w is None:
            return (x_dropped, indices, test_indices), y
        else:
            return (x_dropped, indices, test_indices), y, w


class PrepAEC(object):

    def __init__(
        self,
        offset: int = 1,
        prop_ones: float = 0.8,
        dtype=tf.float32,
        seed=None,
    ):
        import random
        self.offset = offset
        self.prop_ones = prop_ones
        self.dtype = dtype
        self.rng = random.Random(seed)
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
                seed=self.rng.getrandbits(128)
            ),
            dtype=x_dropped.dtype
        )[0]

        indices = tf.range(0, seq_len, dtype=tf.int32)

        # Only include >=0 to avoid real missing data as targets
        test_indices = tf.cast(tf.where(
            tf.logical_and(tf.logical_not(tf.cast(samples, tf.bool)), x >= 0)
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

        if len(tf.shape(x)) == 2:
            indices = tf.repeat(
                tf.expand_dims(indices, 0),
                tf.shape(x)[0],
                axis=0
            )
            test_indices = tf.repeat(
                tf.expand_dims(test_indices, 0),
                tf.shape(x)[0],
                axis=0
            )

        if w is None:
            return (x_dropped, indices, test_indices), y
        else:
            return (x_dropped, indices, test_indices), y, w
