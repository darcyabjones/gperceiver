#!/usr/bin/env python3

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Optional, Union
    from typing import Mapping
    from typing import List
    #  import numpy.typing as npt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Embedding

from tensorflow.keras import initializers, regularizers, constraints

from .layers import (
    SelfAttention,
    CrossAttention,
    SquareRelu,
    AlleleEmbedding
)


class LatentInitialiser(keras.Model):

    def __init__(
        self,
        output_dim: "Optional[int]" = None,
        latent_dim: "Optional[int]" = None,
        initial_values=None,
        latent_initializer=initializers.TruncatedNormal(),
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

        if (output_dim is None) and (latent_dim is None):
            if initial_values is None:
                raise ValueError(
                    "Neither output_dim, latent_dim, nor "
                    "initial_values were specified."
                )

            output_dim = tf.shape(initial_values)[0]
            latent_dim = tf.shape(initial_values)[1]
        else:
            if (output_dim is None) or (latent_dim is None):
                raise ValueError(
                    "Both output_dim and latent_dim need to be specified"
                )

            if initial_values is not None:
                if output_dim != tf.shape(initial_values)[0]:
                    raise ValueError("Shapes aren't the same")

                if latent_dim != tf.shape(initial_values)[1]:
                    raise ValueError("Shapes aren't the same")

        if latent_dim <= 0 or output_dim <= 0:
            raise ValueError(
                'Both `latent_dim` and `output_dim` should be positive, '
                f'Received latent_dim = {latent_dim} and '
                f'output_dim = {output_dim}'
            )

        kwargs["autocast"] = False
        super(LatentInitialiser, self).__init__(**kwargs)

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.initial_values = initial_values

        self.latent_initializer = initializers.get(latent_initializer)
        self.latent_regularizer = regularizers.get(latent_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.latent_constraint = constraints.get(latent_constraint)
        self.supports_masking = False

        self.latent = self.add_weight(
            shape=(self.latent_dim, self.output_dim),
            initializer=self.latent_initializer,
            name='latent',
            regularizer=self.latent_regularizer,
            constraint=self.latent_constraint,
            experimental_autocast=False,
            trainable=self.trainable
        )

        if self.initial_values is not None:
            self.latent.assign(self.initial_values)
        return

    def call(self, X):
        latent = tf.expand_dims(self.latent, 0)
        latent = tf.cast(latent, self.dtype)
        latent = tf.repeat(latent, tf.shape(X)[0], axis=0)
        return latent

    def get_config(self):
        config = super(LatentInitialiser, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "latent_dim": self.latent_dim,
            "initial_values": self.initial_values,
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


"""
class PerceiverBlock(layers.Layer):
    def __init__(
        self,
        projection_units: "Optional[int]" = None,
        num_heads: int = 1,
        num_self_attention: int = 1,
        epsilon: float = 1e-6,
        add_pos: bool = False,
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
        self.add_pos = add_pos

        self.ff_kwargs = ff_kwargs
        if ff_kwargs is None:
            ff_kwargs = {
                "inner_activation": SquareRelu(),
                "epsilon": epsilon,
                "dropout_rate": 0.0
            }

        self.cross_attention_kwargs = cross_attention_kwargs
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        self.cross_attention = CrossAttention(
            projection_units=projection_units,
            name="cross_attention",
            ff_kwargs=ff_kwargs,
            add_pos=add_pos,
            **cross_attention_kwargs
        )

        self.trans_kwargs = trans_kwargs
        if trans_kwargs is None:
            trans_kwargs = {}

        self.transformer_attention = []
        for i in range(num_self_attention):
            i = i + 1
            a = SelfAttention(
                num_heads,
                projection_units=projection_units,
                epsilon=epsilon,
                name=f"transformer_{i}",
                ff_kwargs=ff_kwargs,
                **trans_kwargs
            )
            self.transformer_attention.append(a)

        return

    def call(
        self,
        inputs,
        mask=None,
        training=False,
        return_attention_scores=False
    ):
        # Augment data.
        latent = inputs[0]
        data = inputs[1]
        data_pos = inputs[2] if len(inputs) > 2 else None

        xattentions = []
        sattentions = []

        if return_attention_scores:
            latent, a = self.cross_attention(
                (latent, data, None, data_pos),
                mask=mask,
                training=training,
                return_attention_scores=return_attention_scores
            )
            xattentions.append(a)
        else:
            latent = self.cross_attention(
                (latent, data, None, data_pos),
                mask=mask,
                training=training,
                return_attention_scores=return_attention_scores
            )

        for transformer in self.transformer_attention:
            if return_attention_scores:
                latent, a = transformer(
                    latent,
                    training=training,
                    return_attention_scores=return_attention_scores
                )
                sattentions.appent(a)
            else:
                latent = transformer(
                    latent,
                    training=training,
                    return_attention_scores=False
                )

        if return_attention_scores:
            return latent, xattentions, sattentions
        else:
            return latent

    def get_config(self):
        config = super(PerceiverBlock, self).get_config()
        config.update({
            "projection_units": self.projection_units,
            "num_heads": self.num_heads,
            "num_self_attention": self.num_self_attention,
            "epsilon": self.epsilon,
            "add_pos": self.add_pos,
            "cross_attention_kwargs": self.cross_attention_kwargs,
            "ff_kwargs": self.ff_kwargs,
            "trans_kwargs": self.trans_kwargs
        })
        return config
"""


class PerceiverEncoder(keras.Model):

    def __init__(
        self,
        num_iterations: int,
        projection_units: "Optional[int]" = None,
        num_self_attention: int = 1,
        num_self_attention_heads: int = 2,
        epsilon: float = 1e-6,
        add_pos: bool = False,
        cross_attention_dropout: float = 0.1,
        self_attention_dropout: float = 0.1,
        share_weights: "Union[bool, str]" = "after_first",
        ff_kwargs: "Optional[Mapping[str, Any]]" = None,
        cross_attention_kwargs: "Optional[Mapping[str, Any]]" = None,
        projection_kwargs: "Optional[Mapping[str, Any]]" = None,
        self_attention_kwargs: "Optional[Mapping[str, Any]]" = None,
        **kwargs
    ):
        super(PerceiverEncoder, self).__init__(**kwargs)

        assert num_iterations > 0

        if projection_units is not None:
            assert projection_units > 0

        self.num_iterations = num_iterations
        self.projection_units = projection_units
        self.num_self_attention_heads = num_self_attention_heads
        self.num_self_attention = num_self_attention
        self.epsilon = epsilon
        self.add_pos = add_pos
        self.cross_attention_dropout = cross_attention_dropout
        self.self_attention_dropout = self_attention_dropout
        self.share_weights = share_weights
        assert share_weights in (True, False, "after_first", "after_first_xa")

        if ff_kwargs is not None:
            ff_kwargs = dict(ff_kwargs)
        else:
            ff_kwargs = {
                "inner_activation": SquareRelu(),
                "epsilon": epsilon,
                "dropout_rate": 0.0
            }

        if cross_attention_kwargs is not None:
            cross_attention_kwargs = dict(cross_attention_kwargs)
        else:
            cross_attention_kwargs = {}

        if projection_kwargs is not None:
            projection_kwargs = dict(projection_kwargs)
        else:
            projection_kwargs = {}

        if self_attention_kwargs is not None:
            self_attention_kwargs = dict(self_attention_kwargs)
        else:
            self_attention_kwargs = {}

        self.ff_kwargs = ff_kwargs
        self.cross_attention_kwargs = cross_attention_kwargs
        self.projection_kwargs = projection_kwargs
        self.self_attention_kwargs = self_attention_kwargs
        return

    def gen_xa(self, i: "Union[str,int]"):
        return CrossAttention(
            self.projection_units,
            name=f"cross_attention_{i}",
            epsilon=self.epsilon,
            add_pos=self.add_pos,
            dropout_rate=self.cross_attention_dropout,
            ff_kwargs=self.ff_kwargs,
            projection_kwargs=self.projection_kwargs,
            **self.cross_attention_kwargs
        )

    def gen_sa(self, i: "Union[str,int]", j: "Union[str,int]"):
        return SelfAttention(
            num_heads=self.num_self_attention_heads,
            projection_units=self.projection_units,
            epsilon=self.epsilon,
            dropout_rate=self.self_attention_dropout,
            ff_kwargs=self.ff_kwargs,
            name=f"self_attention_{i}_{j}",
            **self.self_attention_kwargs
        )

    def build(self, input_shape):

        if self.share_weights in ("after_first", "after_first_xa", True):
            shared_sa_blocks = {
                j: self.gen_sa("shared", j)
                for j in range(1, self.num_self_attention + 1)
            }

            def gen_sa(i, j):
                return shared_sa_blocks[j]

            shared_xa_block = self.gen_xa("shared")

            def gen_xa(i):
                return shared_xa_block

            if self.share_weights in ("after_first", "after_first_xa"):
                def gen_first_xa(i):
                    return self.gen_xa("1")
            else:
                gen_first_xa = gen_xa

            if self.share_weights == "after_first_xa":
                gen_first_sa = self.gen_sa
            else:
                gen_first_sa = gen_sa
        else:
            gen_sa = self.gen_sa
            gen_xa = self.gen_xa
            gen_first_sa = self.gen_sa
            gen_first_xa = self.gen_xa

        self.players = [gen_first_xa(1)]

        for j in range(1, self.num_self_attention + 1):
            self.players.append(gen_first_sa(1, j))

        for i in range(2, self.num_iterations + 1):
            self.players.append(gen_xa(i))

            for j in range(1, self.num_self_attention + 1):
                self.players.append(gen_sa(i, j))

        self.built = True
        return

    def call(
        self,
        inputs,
        mask: "Optional[Any]" = None,
        training: "Optional[bool]" = False,
        return_attention_scores: "Optional[bool]" = False
    ):
        latent = inputs[0]
        data = inputs[1]
        data_pos = inputs[2] if len(inputs) > 2 else None

        xattentions: "List[tf.Tensor]" = []
        sattentions: "List[tf.Tensor]" = []

        for layer in self.players:
            if isinstance(layer, CrossAttention):
                if return_attention_scores:
                    latent, a = layer(
                        query=latent,
                        key_value=data,
                        key_pos=data_pos,
                        mask=mask,
                        training=training,
                        return_attention_scores=return_attention_scores
                    )
                    xattentions.append(a)
                else:
                    latent = layer(
                        query=latent,
                        key_value=data,
                        key_pos=data_pos,
                        mask=mask,
                        training=training,
                        return_attention_scores=return_attention_scores
                    )
            elif isinstance(layer, SelfAttention):
                if return_attention_scores:
                    latent, a = layer(
                        latent,
                        training=training,
                        return_attention_scores=return_attention_scores
                    )
                    sattentions.append(a)
                else:
                    latent = layer(
                        latent,
                        training=training,
                        return_attention_scores=False
                    )
            else:
                raise ValueError("Got unexpected layer")

        if return_attention_scores:
            return latent, xattentions, sattentions
        else:
            return latent

    def get_config(self):
        config = super(PerceiverEncoder, self).get_config()
        config["num_iterations"] = self.num_iterations
        config["projection_units"] = self.projection_units
        config["num_self_attention_heads"] = self.num_self_attention_heads
        config["num_self_attention"] = self.num_self_attention
        config["epsilon"] = self.epsilon
        config["add_pos"] = self.add_pos
        config["cross_attention_dropout"] = self.cross_attention_dropout
        config["self_attention_dropout"] = self.self_attention_dropout
        config["share_weights"] = self.share_weights
        config["ff_kwargs"] = self.ff_kwargs
        config["cross_attention_kwargs"] = self.cross_attention_kwargs
        config["projection_kwargs"] = self.projection_kwargs
        config["self_attention_kwargs"] = self.self_attention_kwargs
        return config


class PerceiverEncoderDecoder(keras.Model):

    def __init__(
        self,
        latent_initialiser: LatentInitialiser,
        marker_embedder: "Union[AlleleEmbedding, Embedding]",
        encoder: PerceiverEncoder,
        decoder: CrossAttention,
        predictor: layers.Layer,
        relational_embedder: "Optional[layers.Layer]" = None,
        num_decode_iters: int = 1,
        **kwargs
    ):
        super(PerceiverEncoderDecoder, self).__init__(**kwargs)
        self.latent_initialiser = latent_initialiser
        self.marker_embedder = marker_embedder
        self.encoder = encoder
        self.decoder = decoder
        self.num_decode_iters = num_decode_iters
        self.predictor = predictor
        self.relational_embedder = relational_embedder
        return

    def call(
        self,
        X,
        mask=None,
        return_attention_scores: bool = False,
        training: "Optional[bool]" = False
    ):
        markers, all_pos, test_pos = X

        latent = self.latent_initialiser(markers)
        markers = self.marker_embedder((markers, all_pos))

        if self.relational_embedder is None:
            encoded = self.encoder(
                [latent, markers],
                training=training,
                return_attention_scores=return_attention_scores
            )
        else:
            relational = self.relational_embedder(all_pos)
            encoded = self.encoder(
                [latent, markers, relational],
                training=training,
                return_attention_scores=return_attention_scores
            )

        if return_attention_scores:
            encoded, xattention, sattention = encoded
        else:
            xattention = None
            sattention = None

        preds = self.marker_embedder.position_embedder(test_pos)
        oattention = []
        for _ in range(self.num_decode_iters):
            preds = self.decoder(
                query=preds,
                key_value=encoded,
                training=training,
                return_attention_scores=return_attention_scores
            )

            if return_attention_scores:
                preds, oatt = preds
                oattention.append(oatt)

        preds = self.predictor(preds)

        if return_attention_scores:
            return preds, xattention, sattention, oattention
        else:
            return preds

    def get_config(self):
        config = super(PerceiverEncoderDecoder, self).get_config()
        config["latent_initialiser"] = self.latent_initialiser.get_config()
        config["marker_embedder"] = self.marker_embedder.get_config()
        config["encoder"] = self.marker_encoder.get_config()
        config["decoder"] = self.latent_initializer.get_config()
        config["predictor"] = self.predictor.get_config()
        config["num_decode_iters"] = self.num_decode_iters

        if self.relational_embedder is None:
            config["relational_embedder"] = None
        else:
            config["relational_embedder"] = self.relational_embedder.get_config()  # noqa
        return config


class PerceiverPredictor(object):

    def __init__(self):
        return

    def call(self, X):
        # New latents per env
        # concat latents with new latents
        # encode markers
        # encoder([latent, markers])
        # select env
        # output_dense
        return
