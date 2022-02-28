#!/usr/bin/env python3

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Optional, Union
    from typing import Mapping
    from typing import List
    from typing import Literal
    #  import numpy.typing as npt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Embedding

from tensorflow.keras import initializers, regularizers, constraints

from .layers import (
    SelfAttention,
    CrossAttention,
    AlleleEmbedding,
    AlleleEmbedding2,
    PositionEmbedding,
    FourierPositionEmbedding
)


class LayerWrapper(keras.Model):

    def __init__(self, layer: layers.Layer, **kwargs):
        super(LayerWrapper, self).__init__(**kwargs)
        self.layer = layer
        return

    def call(self, X):
        return self.layer(X)

    def config(self):
        config = super(LayerWrapper, self).get_config()
        config["layer"] = self.layer.get_config()
        return


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
        ff_units: "Optional[int]" = None,
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
        assert share_weights in (True, False, "xa_only", "sa_only",
                                 "after_first", "after_first_xa")

        if ff_kwargs is not None:
            ff_kwargs = dict(ff_kwargs)
        else:
            ff_kwargs = {
                "inner_activation": "gelu",
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
        self.ff_units = ff_units
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
            ff_units=self.ff_units,
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
            ff_units=self.ff_units,
            name=f"self_attention_{i}_{j}",
            **self.self_attention_kwargs
        )

    def build(self, input_shape):  # noqa

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

        elif self.share_weights == "xa_only":
            gen_sa = self.gen_sa
            gen_first_sa = gen_sa

            shared_xa_block = self.gen_xa("shared")

            def gen_xa(i):
                return shared_xa_block

            gen_first_xa = gen_xa

        elif self.share_weights == "sa_only":
            gen_xa = self.gen_xa
            gen_first_xa = gen_xa

            shared_sa_blocks = {
                j: self.gen_sa("shared", j)
                for j in range(1, self.num_self_attention + 1)
            }

            def gen_sa(i, j):
                return shared_sa_blocks[j]

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
        config["num_self_attention"] = self.num_self_attention
        config["num_self_attention_heads"] = self.num_self_attention_heads
        config["epsilon"] = self.epsilon
        config["add_pos"] = self.add_pos
        config["cross_attention_dropout"] = self.cross_attention_dropout
        config["self_attention_dropout"] = self.self_attention_dropout
        config["share_weights"] = self.share_weights
        config["ff_kwargs"] = self.ff_kwargs
        config["ff_units"] = self.ff_units
        config["cross_attention_kwargs"] = self.cross_attention_kwargs
        config["projection_kwargs"] = self.projection_kwargs
        config["self_attention_kwargs"] = self.self_attention_kwargs
        return config


class PerceiverEncoderDecoder(keras.Model):

    def __init__(
        self,
        latent_initialiser: LatentInitialiser,
        position_embedder: "Union[PositionEmbedding, FourierPositionEmbedding, Embedding]",  # noqa
        allele_embedder: "Union[AlleleEmbedding, AlleleEmbedding2, Embedding]",
        encoder: PerceiverEncoder,
        decoder: CrossAttention,
        predictor: layers.Layer,
        relational_embedder: "Optional[layers.Layer]" = None,
        allele_combiner: "Literal['add', 'concat']" = 'add',
        num_decode_iters: int = 1,
        **kwargs
    ):
        super(PerceiverEncoderDecoder, self).__init__(**kwargs)
        self.latent_initialiser = latent_initialiser
        self.position_embedder = position_embedder
        self.allele_embedder = allele_embedder
        self.encoder = encoder
        self.decoder = decoder
        self.num_decode_iters = num_decode_iters
        self.predictor = predictor
        self.relational_embedder = relational_embedder
        self.allele_combiner = allele_combiner

        if allele_combiner == "add":
            self.allele_combiner_layer = layers.Add()
        else:
            self.allele_combiner_layer = layers.Concatenate(axis=-1)
        return

    def call(
        self,
        X,
        mask=None,
        return_attention_scores: bool = False,
        training: "Optional[bool]" = False
    ):
        markers, x_pos, y_pos = X

        latent = self.latent_initialiser(markers)
        x_position_embeddings = self.position_embedder(x_pos)

        if isinstance(self.allele_embedder, layers.Embedding):
            alleles = self.allele_embedder(markers)
            alleles = tf.reduce_sum(alleles, axis=-2)
        elif (
            isinstance(self.allele_embedder, LayerWrapper)
            and isinstance(self.allele_embedder.layer, layers.Embedding)
        ):
            alleles = self.allele_embedder(markers)
            alleles = tf.reduce_sum(alleles, axis=-2)
        else:
            alleles = self.allele_embedder([markers, x_pos])

        markers = self.allele_combiner_layer([alleles, x_position_embeddings])

        if self.relational_embedder is None:
            encoded = self.encoder(
                [latent, markers],
                training=training,
                return_attention_scores=return_attention_scores
            )
        else:
            relational = self.relational_embedder(x_pos)
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

        preds = self.position_embedder(y_pos)
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
        config["position_embedder"] = self.position_embedder.get_config()
        config["allele_embedder"] = self.allele_embedder.get_config()
        config["encoder"] = self.marker_encoder.get_config()
        config["decoder"] = self.latent_initializer.get_config()
        config["predictor"] = self.predictor.get_config()
        config["allele_combiner"] = self.allele_combiner
        config["num_decode_iters"] = self.num_decode_iters

        if self.relational_embedder is None:
            config["relational_embedder"] = None
        else:
            config["relational_embedder"] = self.relational_embedder.get_config()  # noqa
        return config


class TwinnedPerceiverEncoderDecoder(keras.Model):

    def __init__(
        self,
        latent_initialiser: LatentInitialiser,
        position_embedder: "Union[PositionEmbedding, Embedding]",
        allele_embedder: "Union[AlleleEmbedding, AlleleEmbedding2, Embedding]",
        encoder: PerceiverEncoder,
        decoder: CrossAttention,
        allele_predictor: "Optional[layers.Layer]",
        contrast_predictor: layers.Layer,
        relational_embedder: "Optional[layers.Layer]" = None,
        num_decode_iters: int = 1,
        allele_combiner: "Literal['add', 'concat']" = 'add',
        contrast_method: "Literal['add', 'subtract', 'abs_subtract', 'concat']" = "concat",  # noqa
        **kwargs
    ):
        super(TwinnedPerceiverEncoderDecoder, self).__init__(**kwargs)
        self.latent_initialiser = latent_initialiser
        self.position_embedder = position_embedder
        self.allele_embedder = allele_embedder
        self.encoder = encoder
        self.decoder = decoder
        self.num_decode_iters = num_decode_iters
        self.contrast_predictor = contrast_predictor
        self.allele_predictor = allele_predictor
        self.relational_embedder = relational_embedder
        self.allele_combiner = allele_combiner

        if allele_combiner == "add":
            self.allele_combiner_layer = layers.Add()
        else:
            self.allele_combiner_layer = layers.Concatenate(axis=-1)

        self.contrast_method = contrast_method
        return

    def call_single(
        self,
        X,
        latent,
        x_positions,
        x_position_embeddings,
        y_position_embeddings,
        mask=None,
        return_attention_scores: bool = False,
        training: "Optional[bool]" = False
    ):
        if isinstance(self.allele_embedder, layers.Embedding):
            alleles = self.allele_embedder(X)
            alleles = tf.reduce_sum(alleles, axis=-2)
        elif (
            isinstance(self.allele_embedder, LayerWrapper)
            and isinstance(self.allele_embedder.layer, layers.Embedding)
        ):
            alleles = self.allele_embedder(X)
            alleles = tf.reduce_sum(alleles, axis=-2)
        else:
            alleles = self.allele_embedder((X, x_positions))
        markers = self.allele_combiner_layer([alleles, x_position_embeddings])

        if self.relational_embedder is None:
            encoded = self.encoder(
                [latent, markers],
                training=training,
                return_attention_scores=return_attention_scores
            )
        else:
            relational = self.relational_embedder(x_positions)
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

        preds = y_position_embeddings
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

        if return_attention_scores:
            return preds, xattention, sattention, oattention
        else:
            return preds

    def contrast(
        self,
        x1,
        x2
    ):
        method = self.contrast_method
        if method == "add":
            return x1 + x2
        elif method == "subtract":
            return x1 - x2
        elif method == "abs_subtract":
            return tf.abs(x1 - x2)
        elif method == "concat":
            return tf.concat([x1, x2], axis=-1)
        else:
            raise ValueError("Invalid method")
        return

    def call(
        self,
        X,
        mask=None,
        return_attention_scores: bool = False,
        training: "Optional[bool]" = False
    ):
        x1, x2, x_pos, y_pos = X

        latent = self.latent_initialiser(x1)
        x_position_embeddings = self.position_embedder(x_pos)
        y_position_embeddings = self.position_embedder(y_pos)

        encoded1 = self.call_single(
            x1,
            latent,
            x_pos,
            x_position_embeddings,
            y_position_embeddings,
            mask,
            return_attention_scores,
            training
        )
        if return_attention_scores:
            encoded1, xattention1, sattention1, oattention1 = encoded1

        encoded2 = self.call_single(
            x2,
            latent,
            x_pos,
            x_position_embeddings,
            y_position_embeddings,
            mask,
            return_attention_scores,
            training
        )
        if return_attention_scores:
            encoded2, xattention2, sattention2, oattention2 = encoded2

        combined = self.contrast(encoded1, encoded2)
        preds = self.contrast_predictor(combined)

        if self.allele_predictor is None:
            if return_attention_scores:
                return (
                    preds,
                    xattention1, xattention2,
                    sattention1, sattention2,
                    oattention1, oattention2
                )
            else:
                return preds

        preds1 = self.allele_predictor(encoded1)
        preds2 = self.allele_predictor(encoded2)

        if return_attention_scores:
            return (
                preds, preds1, preds2,
                xattention1, xattention2,
                sattention1, sattention2,
                oattention1, oattention2
            )
        else:
            return preds, preds1, preds2

    def get_config(self):
        config = super(TwinnedPerceiverEncoderDecoder, self).get_config()

        if self.allele_predictor is None:
            ap = None
        else:
            ap = self.allele_predictor.get_config()

        config["latent_initialiser"] = self.latent_initialiser.get_config()
        config["position_embedder"] = self.position_embedder.get_config()
        config["allele_embedder"] = self.allele_embedder.get_config()
        config["encoder"] = self.encoder.get_config()
        config["decoder"] = self.decoder.get_config()
        config["contrast_predictor"] = self.contrast_predictor.get_config()
        config["allele_predictor"] = ap
        config["contrast_predictor"] = self.contrast_predictor.get_config()
        config["num_decode_iters"] = self.num_decode_iters
        config["allele_combiner"] = self.allele_combiner
        config["contrast_method"] = self.contrast_method

        if self.relational_embedder is None:
            config["relational_embedder"] = None
        else:
            config["relational_embedder"] = self.relational_embedder.get_config()  # noqa
        return config


class PerceiverPredictor(keras.Model):

    def __init__(
        self,
        nblocks: int,
        latent_output_dim: int,
        latent_initialiser: LatentInitialiser,
        position_embedder: "Union[PositionEmbedding, Embedding]",
        allele_embedder: "Union[AlleleEmbedding, AlleleEmbedding2, Embedding]",
        encoder: PerceiverEncoder,
        predictor: "layers.Layer",
        intercept: "layers.Layer",
        relational_embedder: "Optional[layers.Layer]" = None,
        allele_combiner: "Literal['add', 'concat']" = 'add',
        block_strategy: "Literal['latent', 'pool']" = 'latent',
        **kwargs
    ):
        super(PerceiverPredictor, self).__init__(**kwargs)
        self.nblocks = nblocks
        self.latent_initialiser = latent_initialiser
        self.position_embedder = position_embedder
        self.allele_embedder = allele_embedder
        self.encoder = encoder
        self.predictor = predictor
        self.intercept = intercept
        self.relational_embedder = relational_embedder
        self.allele_combiner = allele_combiner

        if allele_combiner == "add":
            self.allele_combiner_layer = layers.Add()
        else:
            self.allele_combiner_layer = layers.Concatenate(axis=-1)

        self.block_strategy = block_strategy

        if latent_output_dim is not None:
            self.latent_output_dim = latent_output_dim
        elif latent_initialiser.__class__.__name__ == "LatentInitialiser":
            self.latent_output_dim = latent_initialiser.output_dim
        elif (
            hasattr(latent_initialiser, "layer")
            and (
                latent_initialiser.layer.__class__.__name__
                == "LatentInitialiser"
            )
        ):
            self.latent_output_dim = latent_initialiser.layer.output_dim
        else:
            raise ValueError("Couldn't get value of latent_output_dim")

        if block_strategy == 'latent':
            self.block_layer = LatentInitialiser(
                latent_output_dim,
                nblocks,
                name="block_embed",
            )
        else:
            self.block_layer = layers.Embedding(
                input_dim=nblocks,
                output_dim=nblocks * 2,
                name="block_embed"
            )
        return

    def call(
        self,
        X,
        mask=None,
        return_attention_scores: bool = False,
        return_embedding: bool = False,
        training: "Optional[bool]" = False
    ):
        x, x_pos, block = X

        latent = self.latent_initialiser(x, training=training)
        if self.block_strategy == "latent":
            latent = layers.concatenate([
                self.block_layer(x_pos),
                latent
            ], axis=1)

        position_embeddings = self.position_embedder(
            x_pos,
            training=training
        )

        if (
            (self.allele_embedder.__class__.__name__ == "Embedding")
            or (
                hasattr(self.allele_embedder, "layer")
                and (self.allele_embedder.layer.__class__.__name__ == "Embedding")  # noqa
            )
        ):
            alleles = self.allele_embedder(x, training=training)
            alleles = tf.reduce_sum(alleles, axis=-2)
        else:
            alleles = self.allele_embedder((x, x_pos), training=training)

        markers = self.allele_combiner_layer([alleles, position_embeddings])

        if self.relational_embedder is None:
            encoded = self.encoder(
                [latent, markers],
                training=training,
                return_attention_scores=return_attention_scores
            )
        else:
            relational = self.relational_embedder(x_pos)
            encoded = self.encoder(
                [latent, markers, relational],
                training=training,
                return_attention_scores=return_attention_scores
            )

        if return_attention_scores:
            encoded, xattention, sattention = encoded

        if self.block_strategy == "latent":
            embedding = layers.GlobalAveragePooling1D()(tf.squeeze(
                tf.gather(encoded, block, axis=1),
                axis=[2]
            ))
        else:
            embedding = layers.concatenate([
                layers.GlobalAveragePooling1D()(encoded),
                layers.GlobalAveragePooling1D()(
                    self.block_layer(block, training=training)
                ),
            ])

        intercept = layers.GlobalAveragePooling1D()(
            self.intercept(block)
        )
        preds = self.predictor(embedding, training=training)

        preds = preds + intercept

        if return_attention_scores:
            if return_embedding:
                return preds, xattention, sattention, embedding
            else:
                return preds, xattention, sattention

        else:
            if return_embedding:
                return preds, embedding
            else:
                return preds

    def get_config(self):
        config = super(PerceiverPredictor, self).get_config()
        config["nblocks"] = self.nblocks
        config["latent_initialiser"] = self.latent_initialiser.get_config()
        config["position_embedder"] = self.position_embedder.get_config()
        config["allele_embedder"] = self.allele_embedder.get_config()
        config["encoder"] = self.encoder.get_config()
        config["predictor"] = self.predictor.get_config()
        config["intercept"] = self.intercept.get_config()
        config["allele_combiner"] = self.allele_combiner
        config["block_strategy"] = self.block_strategy
        config["latent_output_dim"] = self.latent_output_dim

        if self.relational_embedder is None:
            config["relational_embedder"] = None
        else:
            config["relational_embedder"] = self.relational_embedder.get_config()  # noqa
        return config
