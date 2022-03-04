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


def gen_allele_decoder(ploidy: int = 2, nalleles: int = 3):
    from itertools import combinations_with_replacement
    return list(map(
        list,
        combinations_with_replacement(range(nalleles), ploidy)
    ))


@tf.function
def allele_frequencies(alleles, nalleles):
    # Allele 0 is always assumed to be missing data
    counts = tf.reduce_sum(
        tf.reduce_sum(tf.one_hot(alleles - 1, (nalleles - 1)), axis=-2),
        axis=0
    ) + 1
    freqs = 1 - (counts / tf.reduce_sum(counts, axis=-1, keepdims=True))
    freqs = tf.math.log(freqs)
    freqs = tf.concat([
        tf.fill((tf.shape(freqs)[0], 1), -1.0e+10),
        freqs
    ], axis=-1)
    return freqs


@tf.function
def genotype_prob(alleles, freqs, nalleles):
    return tf.reduce_sum(
        tf.one_hot(alleles, nalleles) * tf.expand_dims(freqs, -2),
        [-1, -2]
    )


class PrepPrediction(object):

    def __init__(
        self,
        allele_decoder: "List[List[int]]" = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]],  # noqa
        ploidy: int = 2,
        dtype=tf.float32,
    ):
        from gperceiver.preprocessing import DecodeAlleles
        self.allele_decoder = DecodeAlleles(
            tf.constant(allele_decoder, dtype=tf.uint8)
        )
        self.ploidy = ploidy
        self.dtype = dtype
        return

    def __call__(self, X):

        def inner(x, b=None, y=None, sample_weights=None):
            if b is None:
                out = [(
                    self.allele_decoder(x),
                    tf.expand_dims(
                        tf.range(0, tf.shape(x)[1], dtype=tf.int64),
                        0
                    ),
                )]
            else:
                out = [(
                    self.allele_decoder(x),
                    tf.expand_dims(
                        tf.range(0, tf.shape(x)[1], dtype=tf.int64),
                        0
                    ),
                    b
                )]

            if y is not None:
                out.append(y)

            if sample_weights is not None:
                assert y is not None
                out.append(sample_weights)
            return tuple(out)

        return (
            X
            .batch(1)
            .map(inner)
            .unbatch()
        )


class PrepData(object):
    def __init__(
        self,
        allele_frequencies: tf.Tensor,
        allele_decoder: "List[List[int]]" = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]],  # noqa
        nalleles: int = 3,
        ploidy: int = 2,
        prop_x: float = 0.5,
        prop_y: float = 0.1,
        frequency_scaler: float = 0.5,
        dtype=tf.float32,
        seed: "Optional[int]" = None,
    ):
        from gperceiver.preprocessing import DecodeAlleles
        self.allele_frequencies = tf.constant(
            allele_frequencies,
            dtype=tf.float32
        )
        allele_decoder = tf.constant(allele_decoder, dtype=tf.uint8)

        self.allele_decoder = DecodeAlleles(allele_decoder)

        self.nalleles = nalleles
        self.ploidy = ploidy
        self.prop_x = prop_x
        self.prop_y = prop_y
        self.dtype = dtype
        self.seed = seed
        self.prep = PrepAEC(
            ploidy=self.ploidy,
            nalleles=self.nalleles,
            allele_frequencies=self.allele_frequencies,
            prop_x=self.prop_x,
            prop_y=self.prop_y,
            dtype=self.dtype,
            frequency_scaler=frequency_scaler,
            seed=self.seed
        )
        return

    def __call__(self, x):
        return (
            x
            .map(self.allele_decoder)
            .map(self.prep)
        )


class PrepContrastData(object):
    def __init__(
        self,
        allele_frequencies: tf.Tensor,
        allele_decoder: "List[List[int]]" = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]],  # noqa
        ploidy: int = 2,
        prop_x: float = 0.5,
        prop_y: float = 0.1,
        dtype=tf.float32,
        seed: "Optional[int]" = None
    ):
        from gperceiver.preprocessing import DecodeAlleles
        self.allele_frequencies = tf.constant(
            allele_frequencies,
            dtype=tf.float32
        )
        self.allele_decoder = DecodeAlleles(
            tf.constant(allele_decoder, dtype=tf.uint8)
        )
        self.ploidy = ploidy
        self.prop_x = prop_x
        self.prop_y = prop_y
        self.dtype = dtype
        self.seed = seed
        self.prep = PrepTwinnedAEC(
            allele_frequencies,
            ploidy=self.ploidy,
            prop_x=self.prop_x,
            prop_y=self.prop_y,
            dtype=self.dtype,
            seed=self.seed
        )
        return

    def __call__(self, x):
        return (
            x
            .map(self.allele_decoder)
            .window(2, stride=1, shift=1)
            .flat_map(lambda z: z)
            .batch(2, drop_remainder=True)
            .map(lambda z: (z[0, ], z[1, ]))
            .map(self.prep)
        )


class DecodeAlleles(object):

    def __init__(self, int_decoder: "tf.Tensor", dtype=tf.dtypes.uint8):
        self.int_decoder = tf.convert_to_tensor(int_decoder, dtype=dtype)
        return

    def __call__(self, X, y=None, w=None):
        decoded = tf.gather(self.int_decoder, X)
        if y is None:
            assert w is None
            return decoded
        else:
            if w is None:
                return decoded, y
            else:
                return decoded, y, w


class AlleleCounts(object):

    def __init__(self, ploidy: int, nalleles: int):
        self.ploidy = ploidy
        self.nalleles = nalleles
        return

    def __call__(self, X, w=None):
        na = self.nalleles - 1
        targets = tf.reduce_sum(tf.one_hot(X - 1, na), axis=-2)
        count_targets = tf.cast(
            (tf.expand_dims(targets, -1) - tf.range(na, dtype=tf.float32)) > 0,
            tf.float32
        )

        shape = tf.shape(X)
        reshaped = tf.reshape(
            count_targets,
            (shape[0], shape[1], na * self.ploidy)
        )

        if w is None:
            return reshaped
        else:
            return reshaped, w


class PrepAEC(object):

    def __init__(
        self,
        ploidy: int,
        nalleles: int,
        allele_frequencies: tf.Tensor,
        prop_x: float = 1.0,
        prop_y: float = 0.1,
        frequency_scaler: float = 0.5,
        dtype=tf.float32,
        seed=None,
    ):
        self.ploidy = ploidy
        self.nalleles = nalleles
        self.allele_frequencies = allele_frequencies
        self.prop_x = prop_x
        self.prop_y = prop_y

        assert 0. < frequency_scaler <= 1.
        self.frequency_scaler = frequency_scaler
        self.dtype = dtype

        if seed is None:
            import random
            self.seed = random.Random()
        else:
            self.seed = random.Random(seed)

        from gperceiver.preprocessing import AlleleCounts
        self.allele_counter = AlleleCounts(ploidy, nalleles)
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

    def sample_positions_single(
        self,
        n: int,
        alleles,
    ):
        alleles = tf.expand_dims(alleles, 0)
        any_missing = tf.reduce_any((alleles == 0), axis=-1)[0]

        probs = (
            self.frequency_scaler *
            genotype_prob(alleles, self.allele_frequencies, self.nalleles)
        )
        # This sets is so that it should be impossible to sample loci with
        # missing values.
        probs = tf.where(any_missing, float("-inf"), probs)
        # Sample twice as many so we can take unique samples
        sample1 = tf.random.categorical(
            logits=tf.cast(probs, tf.float32),
            num_samples=tf.round(2 * n),
            seed=self.seed.getrandbits(128),
        )[0]

        sample2 = tf.unique(sample1).y

        if tf.size(sample2) == n:
            indices = tf.sort(sample2)
        elif tf.size(sample2) < n:
            indices = tf.sort(sample1[:n])
        else:
            indices = tf.sort(sample2[:n])

        return indices

    def __call__(self, X, w=None):
        """Prepares an auto-encoding type task"""
        y = self.count_alleles(X)

        seq_len = tf.cast(tf.shape(X)[0], tf.int64)
        num_to_sample = tf.cast(
            tf.math.round((
                self.prop_y * self.prop_x *
                tf.cast(seq_len, tf.float32)
            )),
            tf.int32
        )

        indices = self.sample_positions_single(
            num_to_sample,
            X
        )

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
        allele_frequencies: tf.Tensor,
        ploidy: int = 2,
        nalleles: int = 3,
        prop_x: float = 1.0,
        prop_y: float = 0.1,
        frequency_scaler: float = 0.5,
        dtype=tf.float32,
        seed=None,
    ):
        self.ploidy = ploidy
        self.nalleles = nalleles
        self.allele_frequencies = allele_frequencies
        self.prop_x = prop_x
        self.prop_y = prop_y

        assert 0. < frequency_scaler <= 1.
        self.frequency_scaler = frequency_scaler
        self.dtype = dtype

        if seed is None:
            import random
            self.seed = random.Random()
        else:
            self.seed = random.Random(seed)

        from gperceiver.preprocessing import AlleleCounts
        self.allele_counter = AlleleCounts(ploidy, nalleles)
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

    def sample_positions_multi(
        self,
        n: int,
        alleles1,
        alleles2,
    ):
        alleles1 = tf.expand_dims(alleles1, 0)
        any_missing1 = tf.reduce_any((alleles1 == 0), axis=-1)

        probs1 = genotype_prob(
            alleles1,
            self.allele_frequencies,
            self.nalleles
        )

        alleles2 = tf.expand_dims(alleles2, 0)
        any_missing2 = tf.reduce_any((alleles2 == 0), axis=-1)

        probs2 = genotype_prob(
            alleles2,
            self.allele_frequencies,
            self.nalleles
        )

        # Probs is in log scale, so adding multiplys
        probs = self.frequency_scaler * (probs1 + probs2)

        any_missing = tf.expand_dims(tf.reduce_any(
            tf.concat([any_missing1, any_missing2], axis=0),
            axis=0
        ), 0)

        # This sets is so that it should be impossible to sample loci with
        # missing values.
        probs = tf.where(any_missing, float("-inf"), probs)
        # Sample twice as many so we can take unique samples
        sample1 = tf.random.categorical(
            logits=tf.cast(probs, tf.float32),
            num_samples=tf.round(2 * n),
            seed=self.seed.getrandbits(128),
        )[0]

        sample2 = tf.unique(sample1).y

        if tf.size(sample2) == n:
            indices = tf.sort(sample2)
        elif tf.size(sample2) < n:
            indices = tf.sort(sample1[:n])
        else:
            indices = tf.sort(sample2[:n])

        return indices

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
            tf.int32
        )

        indices = self.sample_positions_multi(
            num_to_sample,
            X1,
            X2
        )
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
