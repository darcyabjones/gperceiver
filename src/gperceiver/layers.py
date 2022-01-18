#!/usr/bin/env python3


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Optional
    from typing import Dict
    from typing import List
    # import numpy.typing as npt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers, regularizers, constraints


class GetPositionalEncoding(layers.Layer):
    """Computes a standard fourier series sine/cosine positional encoding
    on the fly.

    Keyword arguments:
        positions: A 1D array or tensor, to use instead of a normal range.
            Use if the time series data has an uneven lag.
        output_dim: The number of features to create per position.
            Output shape will be [batch; ntimesteps; output_dim]
        min_freq: A very small float to avoid numerical instability issues.
    """

    def __init__(
        self,
        output_dim: int,
        positions: "Optional[List[float]]" = None,
        min_freq: float = 1e-4,
        **kwargs
    ):
        kwargs["autocast"] = False
        super(GetPositionalEncoding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.min_freq = min_freq

        if positions is None:
            self.positions = None
        else:
            self.positions = tf.convert_to_tensor(positions)

        self.supports_masking = True
        return

    def positional_encoding(self, positions):
        """Thanks to
        https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
        """

        position = tf.cast(positions, dtype=self.dtype)
        mask = tf.range(self.output_dim)
        sin_mask = tf.cast(mask % 2, tf.float32)
        cos_mask = 1 - sin_mask
        exponent = 2 * (mask // 2)
        exponent = (
            tf.cast(exponent, tf.float32) /
            tf.cast(self.output_dim, tf.float32)
        )
        freqs = self.min_freq ** exponent
        angles = tf.einsum('i,j->ij', position, freqs)
        pos_enc = (
            (tf.math.cos(angles) * cos_mask) +
            (tf.math.sin(angles) * sin_mask)
        )
        return pos_enc

    def call(self, X):
        if self.positions is not None:
            pos_enc = self.positional_encoding(self.positions)
        else:
            pos_enc = tf.cast(self.positional_encoding(
                tf.range(start=0, limit=tf.shape(X)[1], delta=1)
            ), dtype=self.dtype)

        pos_enc = tf.expand_dims(pos_enc, 0)
        return tf.repeat(pos_enc, tf.shape(X)[0], axis=0)

    def get_config(self):
        config = super(GetPositionalEncoding, self).get_config()
        config.update({
            "positions": self.positions.numpy().tolist(),
            "output_dim": self.output_dim,
            "min_freq": self.min_freq
        })
        return config


class PrepMarkers(layers.Layer):
    """Computes a standard fourier series sine/cosine positional encoding
    on the fly.

    Keyword arguments:
        positions: A 1D array or tensor, to use instead of a normal range.
            Use if the time series data has an uneven lag.
        output_dim: The number of features to create per position.
            Output shape will be [batch; ntimesteps; output_dim]
        min_freq: A very small float to avoid numerical instability issues.
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        positions: "Optional[List[float]]" = None,
        min_freq: float = 1e-4,
        **kwargs
    ):
        kwargs["autocast"] = False
        super(PrepMarkers, self).__init__(**kwargs)

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.min_freq = min_freq

        if positions is None:
            self.positions = None
        else:
            self.positions = tf.convert_to_tensor(positions)

        self.embed = layers.LocallyConnected1D(
            filters=embed_dim,
            kernel_size=1,
            implementation=3,
            name="embed",
            use_bias=True,
        )

        self.project = layers.Dense(output_dim, name="project")
        self.positional = GetPositionalEncoding(
            output_dim,
            positions=positions,
            min_freq=min_freq,
            name="positional"
        )
        self.add = layers.Add(name="add")

        self.supports_masking = True
        return

    def build(self, input_shape):
        self.input_spec = layers.InputSpec(
            shape=(None, None, input_shape[-1])
        )
        return

    def call(self, X):
        posenc = self.positional(X)

        embedded = self.embed(X)
        projected = self.project(embedded)
        added = self.add([projected, posenc])
        return added

    def get_config(self):
        config = super(PrepMarkers, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "positions": self.positions.numpy().tolist(),
            "output_dim": self.output_dim,
            "min_freq": self.min_freq
        })
        return config


class LearnedLatent(layers.Layer):

    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        latent_initializer='uniform',
        latent_regularizer=None,
        activity_regularizer=None,
        latent_constraint=None,
        **kwargs
    ):
        """ Constructs a 2D tensor of trainable weights.

        Keyword arguments:
          output_dim: the number of "channels" in the embedding.
          latent_dim: the number of learned query vectors.
                      The output matrix will be (batch, latent_dim, output_dim)
        """
        if latent_dim <= 0 or output_dim <= 0:
            raise ValueError(
                'Both `latent_dim` and `output_dim` should be positive, '
                f'Received latent_dim = {latent_dim} and '
                f'output_dim = {output_dim}'
            )

        kwargs["autocast"] = False
        super(LearnedLatent, self).__init__(**kwargs)

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.latent_initializer = initializers.get(latent_initializer)
        self.latent_regularizer = regularizers.get(latent_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.latent_constraint = constraints.get(latent_constraint)
        self.supports_masking = False
        return

    def build(self, input_shape=None):
        self.latent = self.add_weight(
            shape=(self.latent_dim, self.output_dim),
            initializer=self.latent_initializer,
            name='latent',
            regularizer=self.latent_regularizer,
            constraint=self.latent_constraint,
            experimental_autocast=False,
            trainable=True
        )
        self.built = True
        return

    def call(self, inputs, training=False):
        shape = tf.shape(inputs)
        latent = tf.expand_dims(self.latent, 0)
        latent = tf.cast(latent, self.dtype)
        return latent

    def get_config(self):
        config = super(LearnedLatent, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "latent_dim": self.latent_dim,
            'latent_initializer':
                initializers.serialize(self.latent_initializer),
            'latent_regularizer':
                regularizers.serialize(self.latent_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'latent_constraint':
                constraints.serialize(self.latent_constraint),
        })
        return config


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


class CrossAttention(layers.Layer):

    def __init__(
        self,
        projection_units: "Optional[int]" = None,
        dropout_rate: float = 0.1,
        epsilon: float = 1e-6,
        **kwargs
    ):
        """A standard single headed attention layer with
        normalisation and projection layers.
        """

        super(CrossAttention, self).__init__(**kwargs)

        self.projection_units = projection_units
        self.epsilon = epsilon
        self.dropout_rate = dropout_rate

        self.latent_norm = layers.LayerNormalization(
            epsilon=epsilon,
            name="latent_norm"
        )
        self.data_norm = layers.LayerNormalization(
            epsilon=epsilon,
            name="data_norm"
        )

        self.attention = layers.Attention(
            use_scale=True,
            dropout=dropout_rate,
            name="attention"
        )
        self.add = layers.Add(name="add")
        return

    def build(self, input_shape):
        data_shape, group_shape = input_shape
        if self.projection_units is None:
            self.projection_units = group_shape[-1]

        self.query_dense = layers.Dense(
            units=self.projection_units,
            name="query_dense"
        )
        self.key_dense = layers.Dense(
            units=self.projection_units,
            name="key_dense"
        )
        self.value_dense = layers.Dense(
            units=group_shape[-1],
            name="value_dense"
        )
        self.built = True
        return

    def call(self, X, training=False, return_attention_scores=False):
        data, latent = X
        latent = self.latent_norm(latent)
        data = self.data_norm(data)

        query = self.query_dense(latent)
        key = self.key_dense(data)
        value = self.value_dense(data)

        if return_attention_scores:
            attention_output, attention_scores = self.attention(
                [query, value, key],
                training=training,
                return_attention_scores=True
            )
        else:
            attention_output = self.attention(
                [query, value, key],
                training=training,
                return_attention_scores=False
            )
            attention_scores = None

        attention_output = self.add([attention_output, latent])

        if return_attention_scores:
            return attention_output, attention_scores
        else:
            return attention_output

    def get_config(self):
        config = super(CrossAttention, self).get_config()
        config.update({
            "projection_units": self.projection_units,
            "epsilon": self.epsilon,
            "dropout_rate": self.dropout_rate,
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


class TransformerSelfAttention(layers.Layer):

    def __init__(
        self,
        num_heads: int,
        projection_units: "Optional[int]" = None,
        epsilon=1e-6,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(TransformerSelfAttention, self).__init__(**kwargs)

        self.add = layers.Add(name="add")
        self.lnorm = layers.LayerNormalization(epsilon=epsilon, name="lnorm")

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.projection_units = projection_units
        self.epsilon = epsilon
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
        self.built = True
        return

    def call(self, X, training=False, return_attention_scores=False):
        x1 = self.lnorm(X)
        if return_attention_scores:
            attention_output, attention_scores = self.mha(
                x1,
                x1,
                training=training,
                return_attention_scores=return_attention_scores,
                )
        else:
            attention_output = self.mha(
                x1,
                x1,
                training=training,
                return_attention_scores=False,
            )

        x1 = self.add([attention_output, x1])

        if return_attention_scores:
            return x1, attention_scores
        else:
            return x1

    def get_config(self):
        config = super(TransformerSelfAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "projection_units": self.projection_units,
            "epsilon": self.epsilon,
            "dropout_rate": self.dropout_rate
        })
        return config


class PerceiverBlock(layers.Layer):
    def __init__(
        self,
        projection_units: "Optional[int]" = None,
        num_heads: int = 1,
        num_self_attention: int = 1,
        epsilon: float = 1e-6,
        cross_attention_kwargs: "Optional[Dict[str, Any]]" = None,
        ff_kwargs: "Optional[Dict[str, Any]]" = None,
        trans_kwargs: "Optional[Dict[str, Any]]" = None,
        **kwargs
    ):
        super(PerceiverBlock, self).__init__(**kwargs)

        self.projection_units = projection_units
        self.num_heads = num_heads
        self.num_self_attention = num_self_attention
        self.epsilon = epsilon

        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        self.cross_attention_kwargs = cross_attention_kwargs
        self.cross_attention = CrossAttention(
            projection_units=projection_units,
            name="cross_attention",
            **cross_attention_kwargs
        )

        if ff_kwargs is None:
            ff_kwargs = {
                "inner_activation": SquareRelu(),
                "epsilon": epsilon,
                "dropout_rate": 0.5
            }

        self.ff_kwargs = ff_kwargs
        self.ff0 = ResidualDense(
            inner_units=projection_units,
            name="ff_0",
            **ff_kwargs
        )

        if trans_kwargs is None:
            trans_kwargs = {}
        self.trans_kwargs = trans_kwargs

        self.transformer_attention = []
        for i in range(num_self_attention):
            i = i + 1
            a = TransformerSelfAttention(
                num_heads,
                projection_units=projection_units,
                epsilon=epsilon,
                name=f"transformer_{i}",
                **trans_kwargs
            )
            self.transformer_attention.append(a)
            self.transformer_attention.append(ResidualDense(
                inner_units=projection_units,
                name=f"ff_{i}",
                **ff_kwargs
            ))

        return

    def build(self, input_shape):
        X, latent = input_shape
        self.input_spec = [
            layers.InputSpec(shape=(None, None, self.projection_units)),
            layers.InputSpec(shape=(None, None, latent[-1])),
        ]
        self.built = True
        return

    def call(self, X, training=False, return_attention_scores=False):
        # Augment data.
        data, latent = X
        attentions = {}

        if return_attention_scores:
            latent, a = self.cross_attention(
                (data, latent),
                training=training,
                return_attention_scores=return_attention_scores
            )
            attentions[self.cross_attention.name] = a
        else:
            latent = self.cross_attention(
                (data, latent),
                training=training,
                return_attention_scores=return_attention_scores
            )

        latent = self.ff0(latent, training=training)
        for transformer in self.transformer_attention:
            if isinstance(transformer, TransformerSelfAttention):
                if return_attention_scores:
                    latent, a = transformer(
                        latent,
                        training=training,
                        return_attention_scores=return_attention_scores
                    )
                    attentions[transformer.name] = a
                else:
                    latent = transformer(
                        latent,
                        training=training,
                        return_attention_scores=False
                    )
            else:
                latent = transformer(latent, training=training)

        if return_attention_scores:
            return latent, attentions
        else:
            return latent

    def get_config(self):
        config = super(PerceiverBlock, self).get_config()
        config.update({
            "projection_units": self.projection_units,
            "num_heads": self.num_heads,
            "num_self_attention": self.num_self_attention,
            "epsilon": self.epsilon,
            "cross_attention_kwargs": self.cross_attention_kwargs,
            "ff_kwargs": self.ff_kwargs,
            "trans_kwargs": self.trans_kwargs
        })
        return config


class PerceiverDecoderBlock(layers.Layer):
    def __init__(
        self,
        projection_units: "Optional[int]" = None,
        epsilon: float = 1e-6,
        cross_attention_kwargs: "Optional[Dict[str, Any]]" = None,
        ff_kwargs: "Optional[Dict[str, Any]]" = None,
        **kwargs,
    ):
        super(PerceiverDecoderBlock, self).__init__(**kwargs)

        self.projection_units = projection_units
        self.epsilon = epsilon

        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        self.cross_attention_kwargs = cross_attention_kwargs
        self.cross_attention = CrossAttention(
            projection_units=projection_units,
            name="cross_attention",
            **cross_attention_kwargs
        )

        if ff_kwargs is None:
            ff_kwargs = {
                "inner_activation": SquareRelu(),
                "epsilon": epsilon,
                "dropout_rate": 0.5
            }

        self.ff_kwargs = ff_kwargs
        self.ff = ResidualDense(
            inner_units=projection_units,
            name="ff",
            **ff_kwargs
        )
        return

    def call(self, X, training=False, return_attention_scores=False):
        # Augment data.
        data, latent = X
        # Note, the order is reversed to the encoder

        if return_attention_scores:
            data, a = self.cross_attention((latent, data), training=training)
        else:
            data = self.cross_attention((latent, data), training=training)
            a = None
        data = self.ff(data, training=training)

        if return_attention_scores:
            return data, a
        else:
            return data

    def get_config(self):
        config = super(PerceiverDecoderBlock, self).get_config()
        config.update({
            "projection_units": self.projection_units,
            "epsilon": self.epsilon,
            "cross_attention_kwargs": self.cross_attention_kwargs,
            "ff_kwargs": self.ff_kwargs
        })
        return config
