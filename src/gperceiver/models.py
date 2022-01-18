#!/usr/bin/env python3

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional
#     import numpy.typing as npt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from .layers import (
    PerceiverBlock,
    PerceiverDecoderBlock
)


class PerceiverMarkerEncoder(keras.Model):

    def __init__(
        self,
        marker_encoder,
        latent_initializer,
        perceivers,
        num_iterations=1,
        **kwargs
    ):
        super(PerceiverMarkerEncoder, self).__init__(**kwargs)

        self.marker_encoder = marker_encoder
        self.latent_initializer = latent_initializer

        if isinstance(perceivers, PerceiverBlock):
            self.perceivers = [perceivers]
        else:
            self.perceivers = perceivers

        self.num_iterations = num_iterations
        return

    def get_positional_encodings(self, X):
        return self.marker_encoder.positional(X)

    def call(
        self,
        X,
        groups = None,
        training: "Optional[bool]" = False,
        return_attention_scores: "Optional[bool]" = False
    ):
        latent = self.latent_initializer(X)
        if groups is not None:
            if len(tf.shape(groups)) == 2:
                groups = tf.expand_dims(groups, 0)
            if tf.shape(groups)[0] == 1:
                groups = tf.repeat(groups, tf.shape(X)[0], axis=0)
            latent = layers.concatenate([groups, latent], axis=1)

        markers = self.marker_encoder(X)

        attentions = {}
        for i in range(self.num_iterations):
            for perceiver in self.perceivers:
                if return_attention_scores:
                    latent, a = perceiver(
                        [markers, latent],
                        training=training,
                        return_attention_scores=True
                    )
                    attentions.update({
                        f"iter{i}/{k}": v
                        for k, v
                        in a.items()
                    })

                else:
                    latent = perceiver([markers, latent], training=training)

        if return_attention_scores:
            return latent, attentions
        else:
            return latent

    def get_config(self):
        config = super(PerceiverMarkerEncoder, self).get_config()
        config["marker_encoder"] = self.marker_encoder.get_config()
        config["latent_initializer"] = self.latent_initializer.get_config()
        config["perceivers"] = [
            p.get_config()
            for p
            in self.perceivers
        ]
        config["num_iterations"] = self.num_iterations
        return config


class PerceiverEncoderDecoder(keras.Model):

    def __init__(
        self,
        encoder: PerceiverMarkerEncoder,
        decoder: PerceiverDecoderBlock,
        predictor: layers.Layer,
        num_decode_iters: int = 4,
    ):
        super(PerceiverEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_decode_iters = num_decode_iters
        self.predictor = predictor
        return

    def call(
        self,
        X,
        training: "Optional[bool]" = False
    ):
        latent = self.encoder(X, training=False)

        preds = self.encoder.get_positional_encodings(X)
        for _ in range(self.num_decode_iters):
            preds = self.decoder([preds, latent], training=training)
        preds = self.predictor(preds)
        return preds

    def get_config(self):
        config = super(PerceiverEncoderDecoder, self).get_config()
        config["encoder"] = self.marker_encoder.get_config()
        config["decoder"] = self.latent_initializer.get_config()
        config["predictor"] = self.predictor.get_config()
        config["num_decode_iters"] = self.num_decode_iters
        return config
