#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.initializers import Initializer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Optional, List
    import numpy.typing as npt


@tf.function
def positional_encoding(
    positions,
    output_dim,
    min_freq: float = 1e-4,
    dtype=tf.float32
):
    """Thanks to
    https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
    """

    position = tf.cast(positions, dtype=dtype)
    mask = tf.range(output_dim)
    sin_mask = tf.cast(mask % 2, dtype)
    cos_mask = 1 - sin_mask
    exponent = 2 * (mask // 2)
    exponent = (
        tf.cast(exponent, dtype) /
        tf.cast(output_dim, dtype)
    )
    freqs = min_freq ** exponent

    if len(tf.shape(position)) == 2:
        angles = tf.einsum('bi,j->bij', position, freqs)
    else:
        angles = tf.einsum('i,j->ij', position, freqs)

    pos_enc = (
        (tf.math.cos(angles) * cos_mask) +
        (tf.math.sin(angles) * sin_mask)
    )
    return pos_enc


class FourierEncoding(Initializer):

    def __init__(
        self,
        positions: "Optional[Union[tf.Tensor, npt.ArrayLike, List[float]]]" = None,  # noqa
        min_freq: float = 1e-4,
    ):
        if positions is None:
            self.positions = positions
        else:
            self.positions = tf.convert_to_tensor(positions)

        self.min_freq = min_freq
        return

    def __call__(self, shape, dtype=None):

        if self.positions is None:
            positions = tf.range(shape[0])
        else:
            assert tf.math.reduce_all(tf.shape(self.positions)[0] == shape[0])
            positions = self.positions

        return positional_encoding(
            positions,
            output_dim=shape[1],
            min_freq=self.min_freq,
            dtype=dtype
        )

    def get_config(self):
        return {
            'positions': self.positions,
            'min_freq': self.min_freq,
        }


class InitializeWithValues(Initializer):

    def __init__(
        self,
        values: "Union[tf.Tensor, npt.ArrayLike]",
    ):
        self.values = tf.convert_to_tensor(values)
        return

    def __call__(self, shape, dtype=None):
        values = self.values

        assert tf.math.reduce_all(tf.shape(values) == shape)

        if dtype is not None:
            values = tf.cast(values, dtype)

        return values

    def get_config(self):
        return {'values': self.values}
