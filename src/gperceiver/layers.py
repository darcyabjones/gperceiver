#!/usr/bin/env python3

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Optional
    from typing import Dict
    # import numpy.typing as npt

import tensorflow as tf
from tensorflow.keras import layers
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


class CrossAttention(layers.Layer):

    def __init__(
        self,
        projection_units: "Optional[int]" = None,
        dropout_rate: float = 0.1,
        epsilon: float = 1e-6,
        add_pos: bool = False,
        projection_kwargs: "Optional[Dict[str, Any]]" = None,
        ff_kwargs: "Optional[Dict[str, Any]]" = None,
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
                "inner_activation": SquareRelu(),
                "epsilon": self.epsilon,
                "dropout_rate": 0.0
            }
        else:
            ff_kwargs = self.ff_kwargs

        self.ff = ResidualDense(
            inner_units=q_shape[-1],
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
                "inner_activation": SquareRelu(),
                "epsilon": self.epsilon,
                "dropout_rate": 0.0
            }
        else:
            ff_kwargs = self.ff_kwargs

        self.ff = ResidualDense(
            inner_units=input_shape[-1],  # self.projection_units,
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
            "ff_kwargs": self.ff_kwargs
        })
        return config
