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


@tf.function
def prep_aec(x, w=None):
    """Prepares an auto-encoding type task"""

    x_dropped = x + 1

    # Generates a sample of 1 and 0. Most will be 1.
    samples = tf.cast(
        tf.random.categorical(
            tf.math.log([[0.2, 0.8]]),
            tf.shape(x_dropped)[0]
        ),
        dtype=x_dropped.dtype
    )[0]

    x_dropped = samples * x_dropped
    x_dropped = tf.one_hot(x_dropped, 4, dtype=tf.dtypes.float32)
    y = tf.one_hot(x, 3, dtype=tf.dtypes.float32)

    if w is None:
        return x_dropped, y
    else:
        return x_dropped, y, w


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
