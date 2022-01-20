#!/usr/bin/env python3

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional
#     import numpy.typing as npt

import tensorflow.keras as keras
from tensorflow.keras import layers

from .layers import (
    LearnedLatent,
    PerceiverBlock,
    PerceiverDecoderBlock
)


class PerceiverLatentInitialiser(keras.Model):

    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        **kwargs
    ):
        super(PerceiverLatentInitialiser, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.latent_initializer = LearnedLatent(
            output_dim=output_dim,
            latent_dim=latent_dim
        )
        return

    def call(
        self,
        X
    ):
        latent = self.latent_initializer(X)
        return latent

    def get_config(self):
        config = super(PerceiverLatentInitialiser, self).get_config()
        config["output_dim"] = self.output_dim
        config["latent_dim"] = self.latent_dim
        return config


class PerceiverMarkerEncoder(keras.Model):

    def __init__(
        self,
        marker_encoder,
        perceivers,
        num_iterations=1,
        **kwargs
    ):
        super(PerceiverMarkerEncoder, self).__init__(**kwargs)

        self.marker_encoder = marker_encoder

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
        inputs,
        training: "Optional[bool]" = False,
        return_attention_scores: "Optional[bool]" = False
    ):
        X, latent = inputs
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
            i += 1

        if return_attention_scores:
            return latent, attentions
        else:
            return latent

    def get_config(self):
        config = super(PerceiverMarkerEncoder, self).get_config()
        config["marker_encoder"] = self.marker_encoder.get_config()
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
        latent_initialiser: PerceiverLatentInitialiser,
        encoder: PerceiverMarkerEncoder,
        decoder: PerceiverDecoderBlock,
        predictor: layers.Layer,
        num_decode_iters: int = 1,
        **kwargs
    ):
        super(PerceiverEncoderDecoder, self).__init__(**kwargs)
        self.latent_initialiser = latent_initialiser
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
        latent = self.latent_initialiser(X)
        latent = self.encoder([X, latent], training=False)

        preds = self.encoder.get_positional_encodings(X)
        for _ in range(self.num_decode_iters):
            preds = self.decoder([preds, latent], training=training)
        preds = self.predictor(preds)
        return preds

    def get_config(self):
        config = super(PerceiverEncoderDecoder, self).get_config()
        config["latent_initialiser"] = self.latent_initialiser.get_config()
        config["encoder"] = self.marker_encoder.get_config()
        config["decoder"] = self.latent_initializer.get_config()
        config["predictor"] = self.predictor.get_config()
        config["num_decode_iters"] = self.num_decode_iters
        return config
