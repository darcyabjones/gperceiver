#!/usr/bin/env python3

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Optional, Union
    from typing import Dict
    import numpy.typing as npt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.python.keras.layers.dense_attention import BaseDenseAttention


class TwoStepDense(layers.Layer):

    def __init__(
        self,
        units: "Optional[int]" = None,
        inner_units: "Optional[int]" = None,
        activation: "Optional[Any]" = None,
        inner_activation: "Optional[Any]" = None,
        use_bias: bool = True,
        inner_use_bias: bool = True,
        kernel_initializer: "Any" = 'glorot_uniform',
        inner_kernel_initializer: "Any" = 'glorot_uniform',
        bias_initializer: "Any" = 'zeros',
        inner_bias_initializer: "Any" = 'zeros',
        kernel_regularizer: "Any" = None,
        inner_kernel_regularizer: "Any" = None,
        bias_regularizer: "Any" = None,
        inner_bias_regularizer: "Any" = None,
        activity_regularizer: "Any" = None,
        inner_activity_regularizer: "Any" = None,
        kernel_constraint: "Any" = None,
        inner_kernel_constraint: "Any" = None,
        bias_constraint: "Any" = None,
        inner_bias_constraint: "Any" = None,
        **kwargs
    ):
        """ A simple layer that just does low-rank approximations to
        large dense operations.

        E.g. instead of doing a 128 * 128 operation
        (128**2 parameters), you can separate
        it into a 128 * 32 -> 32 * 128, which only uses
        2 * (32 * 128) parameters.

        "inner_" parameters set the size of the middle layer.
        As long as the inner_units is less than half the size of
        the inputs/outputs it will use fewer parameters.
        """

        super(TwoStepDense, self).__init__(**kwargs)

        self.inner_kwargs = {
            "units": inner_units,
            "activation": inner_activation,
            "use_bias": inner_use_bias,
            "kernel_initializer": inner_kernel_initializer,
            "bias_initializer": inner_bias_initializer,
            "kernel_regularizer": inner_kernel_regularizer,
            "bias_regularizer": inner_bias_regularizer,
            "activity_regularizer": inner_activity_regularizer,
            "kernel_constraint": inner_kernel_constraint,
            "bias_constraint": inner_bias_constraint,
        }

        self.projection_kwargs = {
            "units": units,
            "activation": activation,
            "use_bias": use_bias,
            "kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer,
            "kernel_regularizer": kernel_regularizer,
            "bias_regularizer": bias_regularizer,
            "activity_regularizer": activity_regularizer,
            "kernel_constraint": kernel_constraint,
            "bias_constraint": bias_constraint,
        }
        return

    def build(self, input_shape):
        if self.projection_kwargs["units"] is None:
            self.projection_kwargs["units"] = input_shape[-1]

        if self.inner_kwargs["units"] is None:
            from math import ceil
            assert input_shape[-1] > 0
            self.inner_kwargs["units"] = ceil(
                self.projection_kwargs["units"] // 2
            )

        self.inner = layers.Dense(**self.inner_kwargs, name="inner")
        self.project = layers.Dense(**self.projection_kwargs, name="project")
        self.built = True
        return

    def call(self, X):
        X = self.inner(X)
        X = self.project(X)
        return X

    def get_config(self):
        config = super(TwoStepDense, self).get_config()
        config.update(self.projection_kwargs)
        config.update({
            f"inner_{k}": v
            for k, v
            in self.inner_kwargs
        })
        return config


class ResidualDense(layers.Layer):

    def __init__(
        self,
        inner_units: "int",
        inner_activation: "Optional[Any]" = None,
        use_bias: bool = True,
        inner_use_bias: bool = True,
        kernel_initializer: "Any" = 'glorot_uniform',
        inner_kernel_initializer: "Any" = 'glorot_uniform',
        bias_initializer: "Any" = 'zeros',
        inner_bias_initializer: "Any" = 'zeros',
        kernel_regularizer: "Any" = None,
        inner_kernel_regularizer: "Any" = None,
        bias_regularizer: "Any" = None,
        inner_bias_regularizer: "Any" = None,
        activity_regularizer: "Any" = None,
        inner_activity_regularizer: "Any" = None,
        kernel_constraint: "Any" = None,
        inner_kernel_constraint: "Any" = None,
        bias_constraint: "Any" = None,
        inner_bias_constraint: "Any" = None,
        use_layer_norm: bool = True,
        epsilon: float = 1e-6,
        dropout_rate: "Optional[float]" = None,
        **kwargs
    ):
        """A simple dense layer with Residual connection between the input and
        output.
        """

        super(ResidualDense, self).__init__(**kwargs)

        self.use_layer_norm = use_layer_norm
        self.epsilon = epsilon
        if use_layer_norm:
            self.lnorm = layers.LayerNormalization(epsilon=epsilon)

        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = layers.Dropout(dropout_rate)

        self.inner_kwargs = {
            "units": inner_units,
            "activation": inner_activation,
            "use_bias": inner_use_bias,
            "kernel_initializer": inner_kernel_initializer,
            "bias_initializer": inner_bias_initializer,
            "kernel_regularizer": inner_kernel_regularizer,
            "bias_regularizer": inner_bias_regularizer,
            "activity_regularizer": inner_activity_regularizer,
            "kernel_constraint": inner_kernel_constraint,
            "bias_constraint": inner_bias_constraint,
        }

        self.projection_kwargs = {
            "activation": "linear",
            "use_bias": use_bias,
            "kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer,
            "kernel_regularizer": kernel_regularizer,
            "bias_regularizer": bias_regularizer,
            "activity_regularizer": activity_regularizer,
            "kernel_constraint": kernel_constraint,
            "bias_constraint": bias_constraint,
        }

        self.add = layers.Add()
        self.supports_masking = True
        return

    def build(self, input_shape):
        self.projection_kwargs["units"] = input_shape[-1]
        self.inner = layers.Dense(**self.inner_kwargs, name="inner")
        self.project = layers.Dense(**self.projection_kwargs, name="project")
        self.built = True
        return

    def call(self, X):
        if self.use_layer_norm:
            X = self.lnorm(X)
        X0 = self.inner(X)
        X0 = self.project(X0)
        if self.dropout_rate is not None:
            X0 = self.dropout(X0)
        return self.add([X, X0])

    def get_config(self):
        config = super(ResidualDense, self).get_config()
        config.update(self.projection_kwargs)
        config.update({
            f"inner_{k}": v
            for k, v
            in self.inner_kwargs.items()
        })

        config.update({
            "use_layer_norm": self.use_layer_norm,
            "epsilon": self.epsilon,
            "dropout_rate": self.dropout_rate,
        })
        return config


class CustomAttention(BaseDenseAttention):
    def __init__(
        self,
        add_pos=False,
        **kwargs
    ):
        super(CustomAttention, self).__init__(**kwargs)
        self.add_pos = add_pos

        self._custom_built = False
        return

    def custom_build(  # noqa
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
    ):
        """Creates scale variable if use_scale==True."""

        self._custom_built = True

        from tensorflow.python.framework import tensor_shape

        if hasattr(query, "shape"):
            query_shape = tensor_shape.TensorShape(query.shape)
        else:
            query_shape = tensor_shape.TensorShape(query)

        if hasattr(key, "shape"):
            key_shape = tensor_shape.TensorShape(key.shape)
        else:
            key_shape = tensor_shape.TensorShape(key)

        self._key_dim = key_shape[-1]

        # if hasattr(value, "shape"):
        #     value_shape = tensor_shape.TensorShape(value.shape)
        # else:
        #     value_shape = tensor_shape.TensorShape(value)

        if query_pos is None:
            query_pos_shape = None
            self._query_pos_dim = None
        elif hasattr(query_pos, "shape"):
            query_pos_shape = tensor_shape.TensorShape(query_pos.shape)
            self._query_pos_dim = query_pos_shape[-1]
        else:
            query_pos_shape = tensor_shape.TensorShape(query_pos)
            self._query_pos_dim = query_pos_shape[-1]

        if key_pos is None:
            key_pos_shape = None
            self._key_pos_dim = None
        elif hasattr(key_pos, "shape"):
            key_pos_shape = tensor_shape.TensorShape(key_pos.shape)
            self._key_pos_dim = key_pos_shape[-1]
        else:
            key_pos_shape = tensor_shape.TensorShape(key_pos)
            self._key_pos_dim = key_pos_shape[-1]

        if self.add_pos:
            if key_pos_shape is not None:
                assert key_shape == key_pos_shape
            if query_pos_shape is not None:
                assert query_shape == query_pos_shape

        self._custom_built = True

    def _calculate_scores(self, query, key, query_pos=None, key_pos=None):
        """Calculates attention scores as a query-key dot product.
        Args:
          query: Query tensor of shape `[batch_size, Tq, dim]`.
          key: Key tensor of shape `[batch_size, Tv, dim]`.
        Returns:
          Tensor of shape `[batch_size, Tq, Tv]`.
        """

        if self.add_pos:
            if query_pos is not None:
                query += query_pos

            if key_pos is not None:
                key += key_pos

        scores = tf.matmul(query, key, transpose_b=True)
        scores *= 1. / (tf.math.sqrt(float(self._key_dim) + 1e-7))

        if (
            ((query_pos is not None)
             or (key_pos is not None))
            and not self.add_pos
        ):
            if query_pos is None:
                query_pos = query

            if key_pos is None:
                key_pos = key

            pos_scores = tf.matmul(query_pos, key_pos, transpose_b=True)
            pos_scores *= 1. / (tf.math.sqrt(float(self._key_pos_dim) + 1e-7))
            scores += pos_scores

        return scores

    def _validate_call_args(self, inputs, mask):
        """Validates arguments of the call method."""
        class_name = self.__class__.__name__
        if mask is not None:
            if not isinstance(mask, list):
                raise ValueError(
                    '{} layer mask must be a list, '
                    'namely [query_mask, value_mask].'.format(class_name))
            if len(mask) < 2 or len(mask) > len(inputs):
                raise ValueError(
                    (
                        '{} layer mask must be a list of length 2, namely '
                        '[query_mask, value_mask]. Given length: {}'
                    ).format(class_name, len(mask))
                )

    def call(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        mask=None,
        training=None,
        return_attention_scores=False
    ):
        if not self._custom_built:
            self.custom_build(
                query,
                key,
                value,
                query_pos,
                key_pos,
            )

        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(
            query=query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
        )
        if v_mask is not None:
            # Mask of shape [batch_size, 1, Tv].
            v_mask = tf.expand_dims(v_mask, axis=-2)

        if self.causal:
            # Creates a lower triangular mask, so position i cannot attend to
            # positions j>i. This prevents the flow of information from the
            # future into the past.
            scores_shape = tf.shape(scores)
            # causal_mask_shape = [1, Tq, Tv].
            causal_mask_shape = tf.concat(
                [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]],
                axis=0)
            causal_mask = self._lower_triangular_mask(causal_mask_shape)
        else:
            causal_mask = None

        scores_mask = self._merge_masks(v_mask, causal_mask)
        result, attention_scores = self._apply_scores(
            scores=scores,
            value=value,
            scores_mask=scores_mask,
            training=training
        )

        if q_mask is not None:
            # Mask of shape [batch_size, Tq, 1].
            q_mask = tf.expand_dims(q_mask, axis=-1)
            result *= tf.cast(q_mask, dtype=result.dtype)

        if return_attention_scores:
            return result, attention_scores

        return result

    @staticmethod
    def _lower_triangular_mask(shape):
        """Creates a lower-triangular boolean mask over the
        last 2 dimensions.
        """

        row_index = tf.cumsum(
            tf.ones(shape=shape, dtype=tf.int32), axis=-2)
        col_index = tf.cumsum(
            tf.ones(shape=shape, dtype=tf.int32), axis=-1)
        return tf.greater_equal(row_index, col_index)

    @staticmethod
    def _merge_masks(x, y):
        if x is None:
            return y
        if y is None:
            return x
        return tf.logical_and(x, y)

    def get_config(self):
        config = {'use_scale': self.use_scale, "add_pos": self.add_pos}
        base_config = super(CustomAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AlleleEmbedding(layers.Layer):

    def __init__(
        self,
        nalleles: int,
        npositions: int,
        output_dim: int,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        kernel_initializer="uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        activity_regularizer=None,
        activation=None,
        **kwargs,
    ):
        kwargs["autocast"] = False
        super(AlleleEmbedding, self).__init__(
            activity_regularizer=activity_regularizer,
            **kwargs
        )

        self.nalleles = nalleles
        self.npositions = npositions
        self.output_dim = output_dim

        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
        self.activity_regularizer = activity_regularizer
        self.activation = activation

        self.allele_embedder = layers.Embedding(
            nalleles,
            output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=False,
            name="allele_embedder"
        )

        self.kernel = layers.Embedding(
            npositions,
            output_dim * output_dim,
            embeddings_initializer=kernel_initializer,
            embeddings_regularizer=kernel_regularizer,
            embeddings_constraint=kernel_constraint,
            mask_zero=False,
            name="kernel",
        )

        self.reshaper = layers.Reshape((output_dim, output_dim))

        self.bias = layers.Embedding(
            npositions,
            output_dim,
            embeddings_initializer=bias_initializer,
            embeddings_regularizer=bias_regularizer,
            embeddings_constraint=bias_constraint,
            mask_zero=False,
            name="bias"
        )

        self.activation = activations.get(activation)
        return

    def call(self, X):
        alleles, positions = X
        alleles = self.allele_embedder(alleles)
        alleles = tf.reduce_sum(alleles, axis=-2)
        alleles = tf.expand_dims(alleles, -2)

        kernel = self.kernel(positions)
        kshape = tf.shape(kernel)
        kernel = tf.reshape(
            kernel,
            (kshape[0], kshape[1], self.output_dim, self.output_dim)
        )
        bias = self.bias(positions)
        kdota = tf.matmul(alleles, kernel)
        kshape = tf.shape(kdota)
        kdota = tf.reshape(kdota, (kshape[0], kshape[1], kshape[-1]))

        values = kdota + bias
        if self.activation is not None:
            values = self.activation(values)

        return values

    def get_config(self):
        config = super(AlleleEmbedding, self).get_config()
        config.update({
            "nalleles": self.nalleles,
            "npositions": self.npositions,
            "output_dim": self.output_dim,
            "embeddings_initializer": self.embeddings_initializer,
            "embeddings_regularizer": self.embeddings_regularizer,
            "embeddings_constraint": self.embeddings_constraint,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_initializer": self.bias_initializer,
            "bias_regularizer": self.bias_regularizer,
            "bias_constraint": self.bias_constraint,
            "activity_regularizer": self.activity_regularizer,
            "activation": activations.serialize(self.activation),
        })
        return config


class AlleleEmbedding2(layers.Layer):

    def __init__(
        self,
        nalleles: int,
        npositions: int,
        output_dim: int,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        **kwargs,
    ):
        kwargs["autocast"] = False
        super(AlleleEmbedding2, self).__init__(**kwargs)

        self.nalleles = nalleles
        self.npositions = npositions
        self.output_dim = output_dim

        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint

        self.allele_embedder = layers.Embedding(
            nalleles * npositions,
            output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=False,
            name="allele_embedder"
        )
        return

    def calculate_position(self, alleles, positions):
        positions = tf.cast(positions, tf.int64)
        positions = tf.expand_dims(positions, -1)
        nalleles = tf.cast(self.nalleles, tf.int64)
        alleles = tf.cast(alleles, tf.int64)

        return (positions * nalleles) + alleles

    def call(self, X):
        alleles, positions = X
        allele_positions = self.calculate_position(alleles, positions)
        allele_embedding = self.allele_embedder(allele_positions)
        return tf.reduce_sum(allele_embedding, axis=-2)

    def get_config(self):
        config = super(AlleleEmbedding2, self).get_config()
        config.update({
            "nalleles": self.nalleles,
            "npositions": self.npositions,
            "output_dim": self.output_dim,
            "embeddings_initializer": self.embeddings_initializer,
            "embeddings_regularizer": self.embeddings_regularizer,
            "embeddings_constraint": self.embeddings_constraint,
        })
        return config


class PositionEmbedding(layers.Layer):

    def __init__(
        self,
        npositions: int,
        output_dim: int,
        chroms: "Union[tf.Tensor, npt.ArrayLike, None]" = None,
        nchroms: "Optional[int]" = None,
        chrom_output_dim: "Optional[int]" = None,
        position_embeddings_initializer="uniform",
        position_embeddings_regularizer=None,
        position_embeddings_constraint=None,
        position_embeddings_trainable=True,
        position_mask_zero: bool = False,
        chrom_embeddings_initializer="uniform",
        chrom_embeddings_regularizer=None,
        chrom_embeddings_constraint=None,
        chrom_mask_zero: bool = False,
        chrom_embeddings_trainable=True,
        combine_method: str = "add",
        **kwargs,
    ):
        kwargs["autocast"] = False
        super(PositionEmbedding, self).__init__(**kwargs)

        self.npositions = npositions
        self.output_dim = output_dim

        if chroms is None:
            self.chroms = chroms
        else:
            self.chroms = tf.convert_to_tensor(chroms, dtype=tf.int64)

        if self.chroms is None:
            nchroms = None
        else:
            nchroms = tf.size(tf.unique(self.chroms).y).numpy()

        if chrom_output_dim is None:
            chrom_output_dim = output_dim
        self.chrom_output_dim = chrom_output_dim

        if chrom_output_dim != output_dim:
            assert combine_method == "concat"

        self.position_embeddings_initializer = position_embeddings_initializer
        self.position_embeddings_regularizer = position_embeddings_regularizer
        self.position_embeddings_constraint = position_embeddings_constraint
        self.position_mask_zero = position_mask_zero
        self.position_embeddings_trainable = position_embeddings_trainable

        self.chrom_embeddings_initializer = chrom_embeddings_initializer
        self.chrom_embeddings_regularizer = chrom_embeddings_regularizer
        self.chrom_embeddings_constraint = chrom_embeddings_constraint
        self.chrom_mask_zero = chrom_mask_zero
        self.chrom_embeddings_trainable = chrom_embeddings_trainable

        self.combine_method = combine_method

        if combine_method == "add":
            self.combiner = layers.Add(name="combiner")
        elif combine_method == "concat":
            self.combiner = layers.Concatenate(axis=-1, name="combiner")
        else:
            raise ValueError(
                "Combine method must be either 'add' or 'concat'."
            )

        self.position_embedder = layers.Embedding(
            npositions,
            output_dim,
            embeddings_initializer=position_embeddings_initializer,
            embeddings_regularizer=position_embeddings_regularizer,
            embeddings_constraint=position_embeddings_constraint,
            mask_zero=position_mask_zero,
            trainable=position_embeddings_trainable,
            name="position_embedder",
            dtype=self.dtype
        )

        if nchroms is None:
            self.chrom_embedder = None
        else:
            self.chrom_embedder = layers.Embedding(
                nchroms,
                chrom_output_dim,
                embeddings_initializer=chrom_embeddings_initializer,
                embeddings_regularizer=chrom_embeddings_regularizer,
                embeddings_constraint=chrom_embeddings_constraint,
                mask_zero=chrom_mask_zero,
                trainable=chrom_embeddings_trainable,
                name="chrom_embedder",
                dtype=self.dtype
            )
        return

    def call(self, X):

        if self.chrom_embedder is None:
            positions = X
            chroms = None
        else:
            positions = X
            chroms = tf.gather(self.chroms, positions)

        positions = self.position_embedder(positions)

        if (chroms is not None) and (self.chrom_embedder is not None):
            chroms = self.chrom_embedder(chroms)
            return self.combiner([chroms, positions])
        else:
            return positions

    def get_config(self):
        config = super(PositionEmbedding, self).get_config()
        config.update({
            "npositions": self.npositions,
            "output_dim": self.output_dim,
            "chroms": self.chroms.numpy().tolist(),
            "chrom_output_dim": self.chrom_output_dim,
            "position_embeddings_initializer": self.position_embeddings_initializer,  # noqa
            "position_embeddings_regularizer": self.position_embeddings_regularizer,  # noqa
            "position_embeddings_constraint": self.position_embeddings_constraint,  # noqa
            "position_mask_zero": self.position_mask_zero,
            "position_embeddings_trainable": self.position_embeddings_trainable,  # noqa
            "chrom_embeddings_initializer": self.chrom_embeddings_initializer,  # noqa
            "chrom_embeddings_regularizer": self.chrom_embeddings_regularizer,  # noqa
            "chrom_embeddings_constraint": self.chrom_embeddings_constraint,  # noqa
            "chrom_mask_zero": self.chrom_mask_zero,
            "chrom_embeddings_trainable": self.chrom_embeddings_trainable,  # noqa
            "combine_method": self.combine_method,
        })
        return config


class FourierPositionEmbedding(layers.Layer):

    def __init__(
        self,
        positions: "Union[tf.Tensor, npt.ArrayLike]",
        output_dim: int,
        chroms: "Union[tf.Tensor, npt.ArrayLike, None]" = None,
        chrom_output_dim: "Optional[int]" = None,
        position_embeddings_regularizer=None,
        position_embeddings_constraint=None,
        position_embeddings_trainable=False,
        chrom_embeddings_initializer="uniform",
        chrom_embeddings_regularizer=None,
        chrom_embeddings_constraint=None,
        chrom_mask_zero: bool = False,
        chrom_embeddings_trainable=True,
        combine_method: str = "add",
        **kwargs,
    ):
        from .initializers import InitializeWithValues

        kwargs["autocast"] = False
        super(FourierPositionEmbedding, self).__init__(**kwargs)

        self.positions = tf.expand_dims(
            tf.convert_to_tensor(positions, dtype=tf.float32),
            -1
        )
        self.output_dim = output_dim

        if chroms is None:
            self.chroms = chroms
        else:
            self.chroms = tf.convert_to_tensor(chroms, dtype=tf.int64)

        if self.chroms is None:
            nchroms = None
        else:
            nchroms = tf.size(tf.unique(self.chroms).y)

        if chrom_output_dim is None:
            chrom_output_dim = output_dim
        self.chrom_output_dim = chrom_output_dim

        if chrom_output_dim != output_dim:
            assert combine_method == "concat"

        self.position_embeddings_regularizer = position_embeddings_regularizer
        self.position_embeddings_constraint = position_embeddings_constraint
        self.position_embeddings_trainable = position_embeddings_trainable

        self.chrom_embeddings_initializer = chrom_embeddings_initializer
        self.chrom_embeddings_regularizer = chrom_embeddings_regularizer
        self.chrom_embeddings_constraint = chrom_embeddings_constraint
        self.chrom_mask_zero = chrom_mask_zero
        self.chrom_embeddings_trainable = chrom_embeddings_trainable

        self.combine_method = combine_method

        if combine_method == "add":
            self.combiner = layers.Add(name="combiner")
        elif combine_method == "concat":
            self.combiner = layers.Concatenate(axis=-1, name="combiner")
        else:
            raise ValueError(
                "Combine method must be either 'add' or 'concat'."
            )

        self.position_embedder = layers.Embedding(
            tf.size(self.positions),
            1,
            embeddings_initializer=InitializeWithValues(self.positions),
            embeddings_regularizer=position_embeddings_regularizer,
            embeddings_constraint=position_embeddings_constraint,
            mask_zero=False,
            trainable=position_embeddings_trainable,
            name="position_embedder",
            dtype=self.dtype
        )

        if nchroms is None:
            self.chrom_embedder = None
        else:
            self.chrom_embedder = layers.Embedding(
                nchroms,
                chrom_output_dim,
                embeddings_initializer=chrom_embeddings_initializer,
                embeddings_regularizer=chrom_embeddings_regularizer,
                embeddings_constraint=chrom_embeddings_constraint,
                mask_zero=chrom_mask_zero,
                trainable=chrom_embeddings_trainable,
                name="chrom_embedder",
                dtype=self.dtype
            )
        return

    def get_positional_encoding(self, positions):
        from .initializers import positional_encoding
        return positional_encoding(
            positions[:, :, 0],
            self.output_dim,
            min_freq=1e-4,
            dtype=self.dtype
        )

    def call(self, X):

        if self.chrom_embedder is None:
            positions = X
            chroms = None
        else:
            positions = X
            chroms = tf.gather(self.chroms, positions)

        positions = self.position_embedder(positions)
        positions = self.get_positional_encoding(positions)

        if (chroms is not None) and (self.chrom_embedder is not None):
            chroms = self.chrom_embedder(chroms)
            return self.combiner([chroms, positions])
        else:
            return positions

    def get_config(self):
        config = super(FourierPositionEmbedding, self).get_config()
        config.update({
            "positions": self.positions.numpy().tolist(),
            "output_dim": self.output_dim,
            "chroms": self.chroms.numpy().tolist(),
            "chrom_output_dim": self.chrom_output_dim,
            "position_embeddings_initializer": self.position_embeddings_initializer,  # noqa
            "position_embeddings_regularizer": self.position_embeddings_regularizer,  # noqa
            "position_embeddings_constraint": self.position_embeddings_constraint,  # noqa
            "position_embeddings_trainable": self.position_embeddings_trainable,  # noqa
            "chrom_embeddings_initializer": self.chrom_embeddings_initializer,  # noqa
            "chrom_embeddings_regularizer": self.chrom_embeddings_regularizer,  # noqa
            "chrom_embeddings_constraint": self.chrom_embeddings_constraint,  # noqa
            "chrom_mask_zero": self.chrom_mask_zero,
            "chrom_embeddings_trainable": self.chrom_embeddings_trainable,  # noqa
            "combine_method": self.combine_method,
        })
        return config


class CrossAttention(layers.Layer):

    def __init__(
        self,
        projection_units: "Optional[int]" = None,
        dropout_rate: float = 0.1,
        epsilon: float = 1e-6,
        add_pos: bool = False,
        projection_kwargs: "Optional[Dict[str, Any]]" = None,
        ff_kwargs: "Optional[Dict[str, Any]]" = None,
        ff_units: "Optional[int]" = None,
        **kwargs
    ):
        """A standard single headed attention layer with
        normalisation and projection layers.
        """

        super(CrossAttention, self).__init__(**kwargs)

        self.projection_units = projection_units
        self.epsilon = epsilon
        self.dropout_rate = dropout_rate
        self.add_pos = add_pos

        if ff_kwargs is None:
            ff_kwargs = {}
        self.ff_kwargs = ff_kwargs
        self.ff_units = ff_units

        if projection_kwargs is None:
            projection_kwargs = {}
        self.projection_kwargs = projection_kwargs

        self.qnorm = layers.LayerNormalization(
            epsilon=epsilon,
            name="qnorm"
        )
        self.kvnorm = layers.LayerNormalization(
            epsilon=epsilon,
            name="kvnorm"
        )

        self.attention = CustomAttention(
            dropout=dropout_rate,
            name="attention"
        )
        self.add = layers.Add(name="add")

        self.supports_masking = True
        self._custom_built = False
        return

    def custom_build(  # noqa
        self,
        query,
        key_value,
        query_pos=None,
        key_pos=None,
    ):
        from tensorflow.python.framework import tensor_shape
        self._custom_built = True

        if hasattr(query, "shape"):
            q_shape = tensor_shape.TensorShape(query.shape)
        else:
            q_shape = tensor_shape.TensorShape(query)

        # if hasattr(key_value, "shape"):
        #     kv_shape = tensor_shape.TensorShape(key_value.shape)
        # else:
        #     kv_shape = tensor_shape.TensorShape(key_value)

        if query_pos is None:
            query_pos_shape = None
        elif hasattr(query_pos, "shape"):
            query_pos_shape = tensor_shape.TensorShape(query_pos.shape)
        else:
            query_pos_shape = tensor_shape.TensorShape(query_pos)

        if key_pos is None:
            key_pos_shape = None
        elif hasattr(key_pos, "shape"):
            key_pos_shape = tensor_shape.TensorShape(key_pos.shape)
        else:
            key_pos_shape = tensor_shape.TensorShape(key_pos)

        if self.projection_units is None:
            self.projection_units = q_shape[-1]

        self.query_dense = layers.Dense(
            units=self.projection_units,
            name="query_dense",
            **self.projection_kwargs
        )
        self.key_dense = layers.Dense(
            units=self.projection_units,
            name="key_dense",
            **self.projection_kwargs
        )
        self.value_dense = layers.Dense(
            units=q_shape[-1],
            name="value_dense",
            **self.projection_kwargs
        )

        if self.add_pos:
            if query_pos_shape is None:
                self.query_pos_dense = None
            elif query_pos_shape[-1] != self.projection_units:
                self.query_pos_dense = layers.Dense(
                    units=self.projection_units,
                    name="query_pos_dense",
                    **self.projection_kwargs
                )
            else:
                self.query_pos_dense = None

            if key_pos_shape is None:
                self.key_pos_dense = None
            elif key_pos_shape[-1] != self.projection_units:
                self.key_pos_dense = layers.Dense(
                    units=self.projection_units,
                    name="key_pos_dense",
                    **self.projection_kwargs
                )
            else:
                self.key_pos_dense = None
        else:
            if (query_pos_shape is None) and (key_pos_shape is not None):
                self.query_pos_dense = layers.Dense(
                    units=key_pos_shape[-1],
                    name="query_pos_dense",
                    **self.projection_kwargs
                )
            else:
                self.query_pos_dense = None

            if (query_pos_shape is not None) and (key_pos_shape is None):
                self.key_pos_dense = layers.Dense(
                    units=query_pos_shape[-1],
                    name="key_pos_dense",
                    **self.projection_kwargs
                )
            else:
                self.key_pos_dense = None

        if self.ff_kwargs is None:
            ff_kwargs = {
                "inner_activation": "gelu",
                "epsilon": self.epsilon,
                "dropout_rate": 0.0
            }
        else:
            ff_kwargs = self.ff_kwargs

        if self.ff_units is None:
            ff_units = q_shape[-1]
        else:
            ff_units = self.ff_units

        self.ff = ResidualDense(
            inner_units=ff_units,
            name="ff",
            **ff_kwargs
        )
        self._custom_built = True
        return

    def call(
        self,
        query,
        key_value,
        query_pos=None,
        key_pos=None,
        mask=None,
        training=False,
        return_attention_scores=False
    ):
        if not self._custom_built:
            self.custom_build(
                query,
                key_value,
                query_pos,
                key_pos,
            )

        qnorm = self.qnorm(query)
        kvnorm = self.kvnorm(key_value)

        query = self.query_dense(qnorm)
        key = self.key_dense(kvnorm)
        value = self.value_dense(kvnorm)

        if self.query_pos_dense is not None:
            if self.add_pos:
                query_pos = self.query_pos_dense(query_pos)
            else:
                query_pos = self.query_pos_dense(qnorm)
        if self.key_pos_dense is not None:
            if self.add_pos:
                key_pos = self.key_pos_dense(key_pos)
            else:
                key_pos = self.key_pos_dense(kvnorm)

        if return_attention_scores:
            attention_output, attention_scores = self.attention(
                query=query,
                key=key,
                value=value,
                key_pos=key_pos,
                query_pos=query_pos,
                mask=mask,
                training=training,
                return_attention_scores=True
            )
        else:
            attention_output = self.attention(
                query=query,
                key=key,
                value=value,
                key_pos=key_pos,
                query_pos=query_pos,
                mask=mask,
                training=training,
                return_attention_scores=False
            )
            attention_scores = None

        attention_output = self.add([attention_output, qnorm])

        output = self.ff(attention_output, training=training)

        if return_attention_scores:
            return output, attention_scores
        else:
            return output

    def get_config(self):
        config = super(CrossAttention, self).get_config()
        config.update({
            "projection_units": self.projection_units,
            "epsilon": self.epsilon,
            "add_pos": self.add_pos,
            "dropout_rate": self.dropout_rate,
            "projection_kwargs": self.projection_kwargs,
            "ff_kwargs": self.ff_kwargs,
            "ff_units": self.ff_units,
        })
        return config


class SquareRelu(layers.Layer):

    def __init__(self, **kwargs):
        super(SquareRelu, self).__init__(**kwargs)
        self.relu = layers.ReLU()
        return

    def call(self, X):
        X = self.relu(X)
        X = X ** 2
        return X

    def get_config(self):
        config = super(SquareRelu, self).get_config()
        return config


class SelfAttention(layers.Layer):

    def __init__(
        self,
        num_heads: int,
        projection_units: "Optional[int]" = None,
        epsilon: float = 1e-6,
        dropout_rate: float = 0.1,
        ff_kwargs: "Optional[Dict[str, Any]]" = None,
        ff_units: "Optional[int]" = None,
        **kwargs
    ):
        super(SelfAttention, self).__init__(**kwargs)

        self.add = layers.Add(name="add")
        self.lnorm = layers.LayerNormalization(epsilon=epsilon, name="lnorm")

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.projection_units = projection_units
        self.epsilon = epsilon
        self.ff_kwargs = ff_kwargs
        self.ff_units = ff_units
        self.supports_masking = True
        return

    def build(self, input_shape):
        if self.projection_units is None:
            self.projection_units = input_shape[-1]

        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.projection_units,
            value_dim=input_shape[-1],
            dropout=self.dropout_rate,
            name="mha"
        )

        if self.ff_kwargs is None:
            ff_kwargs = {
                "inner_activation": "gelu",
                "epsilon": self.epsilon,
                "dropout_rate": 0.0
            }
        else:
            ff_kwargs = self.ff_kwargs

        if self.ff_units is None:
            ff_units = input_shape[-1]
        else:
            ff_units = self.ff_units

        self.ff = ResidualDense(
            inner_units=ff_units,  # self.projection_units,
            name="ff",
            **ff_kwargs
        )
        self.built = True
        return

    def call(
        self,
        X,
        mask=None,
        training=False,
        return_attention_scores=False
    ):
        x1 = self.lnorm(X)
        if return_attention_scores:
            attention_output, attention_scores = self.mha(
                x1,
                x1,
                attention_mask=mask,
                training=training,
                return_attention_scores=return_attention_scores,
                )
        else:
            attention_output = self.mha(
                x1,
                x1,
                attention_mask=mask,
                training=training,
                return_attention_scores=False,
            )

        x1 = self.add([attention_output, x1])
        output = self.ff(x1, training=training)

        if return_attention_scores:
            return output, attention_scores
        else:
            return output

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "projection_units": self.projection_units,
            "epsilon": self.epsilon,
            "dropout_rate": self.dropout_rate,
            "ff_kwargs": self.ff_kwargs,
            "ff_units": self.ff_units
        })
        return config
