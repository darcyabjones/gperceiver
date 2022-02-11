from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List
    from typing import Optional

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


class PrepData(object):
    def __init__(
        self,
        allele_decoder: "List[List[int]]" = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]],  # noqa
        ploidy: int = 2,
        prop_x: float = 0.5,
        prop_y: float = 0.1,
        dtype=tf.float32,
        seed: "Optional[int]" = None,
    ):
        from gperceiver.preprocessing import DecodeAlleles
        self.allele_decoder = DecodeAlleles(
            tf.constant(allele_decoder, dtype=tf.uint8)
        )
        self.ploidy = ploidy
        self.prop_x = prop_x
        self.prop_y = prop_y
        self.dtype = dtype
        self.seed = seed
        return

    def __call__(self, x):
        return (
            x
            .batch(1)
            .map(self.allele_decoder)
            .unbatch()
            .map(PrepAEC(
                ploidy=self.ploidy,
                prop_x=self.prop_x,
                prop_y=self.prop_y,
                dtype=self.dtype,
                seed=self.seed
            ))
        )


class PrepContrastData(object):
    def __init__(
        self,
        allele_decoder: "List[List[int]]" = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]],  # noqa
        ploidy: int = 2,
        prop_x: float = 0.5,
        prop_y: float = 0.1,
        dtype=tf.float32,
        seed: "Optional[int]" = None
    ):
        from gperceiver.preprocessing import DecodeAlleles
        self.allele_decoder = DecodeAlleles(
            tf.constant(allele_decoder, dtype=tf.uint8)
        )
        self.ploidy = ploidy
        self.prop_x = prop_x
        self.prop_y = prop_y
        self.dtype = dtype
        self.seed = seed
        return

    def __call__(self, x):
        return (
            x
            .batch(1)
            .map(self.allele_decoder)
            .unbatch()
            .window(2, stride=1, shift=1)
            .flat_map(lambda z: z)
            .batch(2, drop_remainder=True)
            .map(lambda z: (z[0, ], z[1, ]))
            .map(PrepTwinnedAEC(
                ploidy=self.ploidy,
                prop_x=self.prop_x,
                prop_y=self.prop_y,
                dtype=self.dtype,
                seed=self.seed
            ))
        )


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
        ploidy: int,
        prop_x: float = 1.0,
        prop_y: float = 0.1,
        dtype=tf.float32,
        seed=None,
    ):
        self.ploidy = ploidy
        self.prop_x = prop_x
        self.prop_y = prop_y
        self.dtype = dtype
        if seed is None:
            import random
            self.seed = random.Random()
        else:
            self.seed = random.Random(seed)

        from gperceiver.preprocessing import AlleleCounts
        self.allele_counter = AlleleCounts()
        return

    @staticmethod
    def ones_from_indices(indices, seqlen, dtype=tf.float32):
        if len(tf.shape(indices)) == 1:
            indices = tf.expand_dims(indices, -1)

        indices = tf.cast(indices, tf.int64)

        zeros = tf.zeros_like(indices)
        indices = tf.concat([zeros, tf.sort(indices, axis=0)], -1)
        del zeros

        length = tf.shape(indices)[0]
        sp = tf.sparse.SparseTensor(
            indices,
            tf.ones((length,), dtype=dtype),
            (1, seqlen)
        )
        return tf.sparse.to_dense(sp)[0]

    def count_diffs(self, y1, y2):

        targets = y1 * y2
        targets = tf.reduce_sum(targets, axis=-1)

        shape = tf.shape(targets)
        targets = tf.cast(
            (
                tf.expand_dims(targets, -1) -
                tf.range(self.ploidy, dtype=tf.float32)
            ) > 0,
            tf.float32
        )

        return tf.reshape(targets, (shape[0], self.ploidy))

    def count_alleles(self, X):
        X = tf.expand_dims(X, 0)
        return self.allele_counter(X)[0]

    def __call__(self, X, w=None):
        """Prepares an auto-encoding type task"""
        y = self.count_alleles(X)

        seq_len = tf.cast(tf.shape(X)[0], tf.int64)
        num_to_sample = tf.cast(
            tf.math.round((
                self.prop_y * self.prop_x *
                tf.cast(seq_len, tf.float32)
            )),
            tf.int64
        )

        is_null = tf.reduce_sum(y, axis=-1) == 0

        candidates = tf.where(tf.logical_not(is_null))[:, 0]
        candidates = tf.random.shuffle(
            candidates,
            seed=self.seed.getrandbits(128)
        )
        ncandidates = tf.size(candidates, out_type=tf.int64)

        if ncandidates < (num_to_sample * 2):
            new_num_to_sample = tf.cast(
                tf.math.round(ncandidates / 2),
                tf.int64
            )
            remaining = num_to_sample - new_num_to_sample
            num_to_sample = new_num_to_sample
            del new_num_to_sample
        else:
            remaining = tf.cast(0, tf.int64)

        indices = tf.gather(candidates, tf.range(num_to_sample))

        if remaining > 0:
            empty_candidates = tf.where(is_null)[:, 0]
            empty_candidates = tf.random.shuffle(
                empty_candidates,
                seed=self.seed.getrandbits(128)
            )
            empty_candidates = tf.gather(empty_candidates, tf.range(remaining))
            indices = tf.concat([indices, empty_candidates], 0)

        indices = tf.sort(indices, axis=0)
        mask = 1 - self.ones_from_indices(indices, seq_len, dtype=X.dtype)
        mask = tf.expand_dims(mask, -1)
        X = X * mask

        xnum_to_sample = tf.cast(
            tf.math.round((self.prop_x * tf.cast(seq_len, tf.float32))),
            tf.int64
        )

        xindices = tf.range(0, seq_len, dtype=indices.dtype)
        if xnum_to_sample < seq_len:
            xindices = tf.random.shuffle(
                xindices,
                seed=self.seed.getrandbits(128)
            )
            xindices = tf.gather(xindices, tf.range(xnum_to_sample))
            xindices = tf.sort(xindices)

        X = tf.gather(X, xindices, axis=0)

        y = tf.gather(y, indices, axis=-2)

        if w is None:
            return (X, xindices, indices), y
        else:
            return (X, xindices, indices), y, w


class PrepTwinnedAEC(object):

    def __init__(
        self,
        ploidy: int,
        prop_x: float = 1.0,
        prop_y: float = 0.1,
        dtype=tf.float32,
        seed=None,
    ):
        self.ploidy = ploidy
        self.prop_x = prop_x
        self.prop_y = prop_y
        self.dtype = dtype
        if seed is None:
            import random
            self.seed = random.Random()
        else:
            self.seed = random.Random(seed)

        from gperceiver.preprocessing import AlleleCounts
        self.allele_counter = AlleleCounts()
        return

    @staticmethod
    def ones_from_indices(indices, seqlen, dtype=tf.float32):
        if len(tf.shape(indices)) == 1:
            indices = tf.expand_dims(indices, -1)

        indices = tf.cast(indices, tf.int64)

        zeros = tf.zeros_like(indices)
        indices = tf.concat([zeros, tf.sort(indices, axis=0)], -1)
        del zeros

        length = tf.shape(indices)[0]
        sp = tf.sparse.SparseTensor(
            indices,
            tf.ones((length,), dtype=dtype),
            (1, seqlen)
        )
        return tf.sparse.to_dense(sp)[0]

    def count_diffs(self, y1, y2):

        targets = y1 * y2
        targets = tf.reduce_sum(targets, axis=-1)

        shape = tf.shape(targets)
        targets = tf.cast(
            (
                tf.expand_dims(targets, -1) -
                tf.range(self.ploidy, dtype=tf.float32)
            ) > 0,
            tf.float32
        )

        return tf.reshape(targets, (shape[0], self.ploidy))

    def count_alleles(self, X):
        X = tf.expand_dims(X, 0)
        return self.allele_counter(X)[0]

    def __call__(self, X1, X2, w=None):
        """Prepares an auto-encoding type task"""
        y1 = self.count_alleles(X1)
        y2 = self.count_alleles(X2)

        seq_len = tf.cast(tf.shape(X1)[0], tf.int64)
        num_to_sample = tf.cast(
            tf.math.round((
                self.prop_y * self.prop_x *
                tf.cast(seq_len, tf.float32) / 2
            )),
            tf.int64
        )

        any_different = tf.reduce_any(y1 != y2, axis=-1)
        is_null = tf.logical_or(
            tf.reduce_sum(y1, axis=-1) == 0,
            tf.reduce_sum(y2, axis=-1) == 0
        )

        diff_candidates = tf.where(tf.logical_and(
            any_different,
            tf.logical_not(is_null)
        ))[:, 0]
        same_candidates = tf.where(tf.logical_and(
            tf.logical_not(any_different), tf.logical_not(is_null)
        ))[:, 0]

        diff_candidates = tf.random.shuffle(
            diff_candidates,
            seed=self.seed.getrandbits(128)
        )
        same_candidates = tf.random.shuffle(
            same_candidates,
            seed=self.seed.getrandbits(128)
        )

        if tf.size(diff_candidates, out_type=tf.int64) < (num_to_sample * 2):
            num_diff = tf.cast(
                tf.math.round(len(diff_candidates) / 2),
                tf.int64
            )
        else:
            num_diff = num_to_sample

        diff_indices = tf.gather(diff_candidates, tf.range(num_diff))
        remaining = num_to_sample - num_diff

        if (
            tf.size(same_candidates, out_type=tf.int64)
            < ((num_to_sample + remaining) * 2)
        ):
            num_same = tf.cast(
                tf.math.round(len(same_candidates) / 2),
                tf.int64
            )
        else:
            num_same = num_to_sample + remaining

        same_indices = tf.gather(
            same_candidates,
            tf.range(num_same),
            name="same"
        )
        indices = tf.concat([diff_indices, same_indices], 0)
        remaining = (num_to_sample * 2) - (num_diff + num_same)

        if remaining > 0:
            extras = tf.random.uniform(
                (remaining,),
                minval=0,
                maxval=seq_len,
                dtype=same_indices.dtype
            )

            indices = tf.concat([indices, extras], 0)
            del extras

        indices = tf.sort(indices, axis=0)
        mask = 1 - self.ones_from_indices(indices, seq_len, dtype=X1.dtype)
        mask = tf.expand_dims(mask, -1)
        X1 = X1 * mask
        X2 = X2 * mask

        xnum_to_sample = tf.cast(
            tf.math.round((self.prop_x * tf.cast(seq_len, tf.float32))),
            tf.int64
        )

        xindices = tf.range(0, seq_len, dtype=indices.dtype)
        if xnum_to_sample < seq_len:
            xindices = tf.random.shuffle(
                xindices,
                seed=self.seed.getrandbits(128)
            )
            xindices = tf.gather(xindices, tf.range(xnum_to_sample))
            xindices = tf.sort(xindices)

        X1 = tf.gather(X1, xindices, axis=0)
        X2 = tf.gather(X2, xindices, axis=0)

        y1 = tf.gather(y1, indices, axis=-2)
        y2 = tf.gather(y2, indices, axis=-2)

        y = self.count_diffs(y1, y2)

        if w is None:
            return (X1, X2, xindices, indices), (y, y1, y2)
        else:
            return (X1, X2, xindices, indices), (y, y1, y2), w
