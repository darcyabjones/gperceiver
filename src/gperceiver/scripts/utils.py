#!/usr/bin/env python3

from typing import TYPE_CHECKING

from tensorflow import keras
from tensorflow.keras import layers

if TYPE_CHECKING:
    from typing import Any, Optional, Union
    from typing import Literal
    from typing import Sequence, List
    from typing import Mapping, Dict

from ..functional import or_else

from ..layers import (
    AlleleEmbedding,
    AlleleEmbedding2,
    CrossAttention,
    PositionEmbedding,
    FourierPositionEmbedding
)

from ..models import (
    LayerWrapper,
    LatentInitialiser,
    PerceiverEncoder,
    PerceiverEncoderDecoder,
    TwinnedPerceiverEncoderDecoder,
)

from ..preprocessing import gen_allele_decoder


class Params(object):

    def __init__(
        self,
        nmarkers: int,
        chrom_pos: "Sequence[int]",
        marker_pos: "Sequence[float]",
        nalleles: "Optional[int]" = None,
        ploidy: "Optional[int]" = None,
        allele_embed_kind: "Optional[Literal['1', '2', '3']]" = None,
        allele_embed_dim: "Optional[int]" = None,
        allele_combiner: "Optional[Literal['add', 'concat']]" = None,
        position_embed_kind: "Optional[Literal['random', 'fourier']]" = None,
        position_embed_dim: "Optional[int]" = None,
        position_embed_trainable: "Optional[bool]" = None,
        share_weights: "Union[None, bool, Literal['xa_only', 'sa_only', 'after_first_xa', 'after_first']]" = None,  # noqa
        projection_dim: "Optional[int]" = None,
        feedforward_dim: "Optional[int]" = None,
        latent_dim: "Optional[int]" = None,
        output_dim: "Optional[int]" = None,
        num_self_attention_heads: "Optional[int]" = None,
        num_self_attention: "Optional[int]" = None,
        num_encode_iters: "Optional[int]" = None,
        num_decode_iters: "Optional[int]" = None,
        contrastive: "Optional[bool]" = None,
        contrastive_weight: "Optional[float]" = None,
        nblocks: "Optional[int]" = None,
        block_strategy: "Optional[Literal['pool', 'latent']]" = None,
        allele_decoder: "Optional[List[List[int]]]" = None,
        **kwargs
    ):
        self.nmarkers: int = int(nmarkers)
        self.chrom_pos: "List[int]" = list(map(int, chrom_pos))
        self.marker_pos: "List[float]" = list(map(float, marker_pos))
        self.nalleles: int = int(or_else(3, nalleles))
        self.ploidy: int = int(or_else(2, ploidy))

        self.allele_embed_kind: "Literal['1', '2', '3']" = or_else(
            '1',
            allele_embed_kind,
        )
        self.__assert_options(
            "allele_embed_kind",
            ['1', '2', '3']
        )

        self.allele_embed_dim: int = int(or_else(256, allele_embed_dim))

        self.allele_combiner: "Literal['add', 'concat']" = or_else(
            'add',
            allele_combiner,
        )
        self.__assert_options(
            "allele_combiner",
            ['add', 'concat']
        )

        self.position_embed_kind: "Literal['random', 'fourier']" = or_else(
            'random',
            position_embed_kind,
        )
        self.__assert_options(
            "position_embed_kind",
            ['random', 'fourier']
        )

        self.position_embed_dim: int = int(or_else(256, position_embed_dim))
        self.position_embed_trainable: bool = bool(or_else(
            True,
            position_embed_trainable,
        ))

        self.share_weights: "Union[bool, Literal['xa_only', 'sa_only', 'after_first_xa', 'after_first']]" = or_else(  # noqa
            True,
            share_weights,
        )
        self.__assert_options(
            "share_weights",
            [True, False, 'xa_only', 'sa_only',
             'after_first_xa', 'after_first']
        )

        self.projection_dim: int = int(or_else(256, projection_dim))
        self.feedforward_dim: int = int(or_else(512, feedforward_dim))
        self.latent_dim: int = int(or_else(256, latent_dim))
        self.output_dim: int = int(or_else(512, output_dim))
        self.num_self_attention_heads: int = int(or_else(4, num_self_attention_heads))  # noqa
        self.num_self_attention: int = int(or_else(2, num_self_attention))
        self.num_encode_iters: int = int(or_else(4, num_encode_iters))
        self.num_decode_iters: int = int(or_else(2, num_decode_iters))
        self.contrastive: bool = bool(or_else(True, contrastive))

        self.contrastive_weight: float = float(or_else(0.0, contrastive_weight))  # noqa
        self.nblocks: "Optional[int]" = nblocks
        self.block_strategy: "Literal['pool', 'latent']" = or_else(
            "pool",
            block_strategy,
        )
        self.__assert_options("block_strategy", ["pool", "latent"])

        self.allele_decoder: "List[List[int]]" = or_else(
            gen_allele_decoder(self.ploidy, self.nalleles),
            allele_decoder,
        )

        self.__check_options()
        return

    def __assert_options(self, attr: str, options: 'Sequence[Any]'):
        val = getattr(self, attr)
        if val not in options:
            raise ValueError(
                f"Got invalid value to argument {attr}. "
                f"Got {val}, expected one of {options}."
            )
        return

    def __check_options(self):
        if self.allele_combiner == "add":
            assert self.allele_embed_dim == self.position_embed_dim
        return

    @classmethod
    def from_dict(
        cls,
        params: "Mapping[str, Any]",
        nmarkers: "Optional[int]" = None,
        chrom_pos: "Optional[Sequence[int]]" = None,
        marker_pos: "Optional[Sequence[float]]" = None,
        nalleles: "Optional[int]" = None,
        ploidy: "Optional[int]" = None,
        allele_embed_kind: "Optional[Literal['1', '2', '3']]" = None,
        allele_embed_dim: "Optional[int]" = None,
        allele_combiner: "Optional[Literal['add', 'concat']]" = None,
        position_embed_kind: "Optional[Literal['random', 'fourier']]" = None,
        position_embed_dim: "Optional[int]" = None,
        position_embed_trainable: "Optional[bool]" = None,
        share_weights: "Union[None, bool, Literal['xa_only', 'sa_only', 'after_first_xa', 'after_first']]" = None,  # noqa
        projection_dim: "Optional[int]" = None,
        feedforward_dim: "Optional[int]" = None,
        latent_dim: "Optional[int]" = None,
        output_dim: "Optional[int]" = None,
        num_self_attention_heads: "Optional[int]" = None,
        num_self_attention: "Optional[int]" = None,
        num_encode_iters: "Optional[int]" = None,
        num_decode_iters: "Optional[int]" = None,
        contrastive: "Optional[bool]" = None,
        contrastive_weight: "Optional[float]" = None,
        nblocks: "Optional[int]" = None,
        block_strategy: "Optional[Literal['pool', 'latent']]" = None,
        allele_decoder: "Optional[List[List[int]]]" = None,
    ):
        if nmarkers is None:
            assert "nmarkers" in params
            nmarkers_: int = int(params["nmarkers"])
        else:
            nmarkers_ = nmarkers

        if chrom_pos is None:
            assert "chrom_pos" in params
            chrom_pos_: "Sequence[int]" = list(map(int, params["chrom_pos"]))
        else:
            chrom_pos_ = chrom_pos

        if marker_pos is None:
            assert "marker_pos" in params
            marker_pos_: "Sequence[float]" = list(map(
                float,
                params["marker_pos"]
            ))
        else:
            marker_pos_ = marker_pos
        cw = or_else(params.get("contrastive_weight", None), contrastive_weight)  # noqa

        return cls(
            nmarkers_,
            chrom_pos_,
            marker_pos_,
            or_else(params.get("nalleles", None), nalleles),
            or_else(params.get("ploidy", None), ploidy),
            or_else(params.get("allele_embed_kind", None), allele_embed_kind),
            or_else(params.get("allele_embed_dim", None), allele_embed_dim),
            or_else(params.get("allele_combiner", None), allele_combiner),
            or_else(params.get("position_embed_kind", None), position_embed_kind),  # noqa
            or_else(params.get("position_embed_dim", None), position_embed_dim),  # noqa
            or_else(params.get("position_embed_trainable", None), position_embed_trainable),  # noqa
            or_else(params.get("share_weights", None), share_weights),
            or_else(params.get("projection_dim", None), projection_dim),
            or_else(params.get("feedforward_dim", None), feedforward_dim),
            or_else(params.get("latent_dim", None), latent_dim),
            or_else(params.get("output_dim", None), output_dim),
            or_else(params.get("num_self_attention_heads", None), num_self_attention_heads),  # noqa
            or_else(params.get("num_self_attention", None), num_self_attention),  # noqa
            or_else(params.get("num_encode_iters", None), num_encode_iters),
            or_else(params.get("num_decode_iters", None), num_decode_iters),
            or_else(params.get("contrastive", None), contrastive),
            cw,
            or_else(params.get("nblocks", None), nblocks),  # noqa
            or_else(params.get("block_strategy", None), block_strategy),  # noqa
            or_else(params.get("allele_decoder", None), allele_decoder),  # noqa
        )

    def to_dict(self) -> "Dict[str, Any]":
        attrs = [
            'nmarkers',
            'chrom_pos',
            'marker_pos',
            'nalleles',
            'ploidy',
            'allele_embed_kind',
            'allele_embed_dim',
            'allele_combiner',
            'position_embed_kind',
            'position_embed_dim',
            'position_embed_trainable',
            'share_weights',
            'projection_dim',
            'feedforward_dim',
            'latent_dim',
            'output_dim',
            'num_self_attention_heads',
            'num_self_attention',
            'num_encode_iters',
            'num_decode_iters',
            'contrastive',
            'contrastive_weight',
            'nblocks',
            'block_strategy',
            'allele_decoder',
        ]
        return {a: getattr(self, a) for a in attrs}


def build_encoder_model(
    params: Params,
):
    # This stuff initialises it so that the encoder can take
    # variable length sequences
    if params.allele_combiner == "add":
        data_dims = params.allele_embed_dim
    else:
        data_dims = params.allele_embed_dim + params.position_embed_dim

    lsize = [
        layers.Input((None, params.output_dim)),
        layers.Input((None, data_dims)),
    ]

    latent_initialiser = LatentInitialiser(
        params.output_dim,
        params.latent_dim,
        name="latent"
    )

    if params.position_embed_kind == "fourier":
        position_embedding_ = FourierPositionEmbedding(
            positions=params.marker_pos,
            chroms=params.chrom_pos,
            output_dim=params.position_embed_dim,
            position_embeddings_trainable=params.position_embed_trainable,
            name="position_embedding",
        )
    else:
        position_embedding_ = PositionEmbedding(
            npositions=params.nmarkers,
            chroms=params.chrom_pos,
            output_dim=params.position_embed_dim,
            position_embeddings_trainable=params.position_embed_trainable,
            name="position_embedding",
        )

    position_embedding = LayerWrapper(
        position_embedding_,
        name="position_embedding_model"
    )

    if params.allele_embed_kind == "2":
        allele_embedding_ = AlleleEmbedding(
            nalleles=params.nalleles,
            npositions=params.nmarkers,
            output_dim=params.allele_embed_dim,
            name="allele_embedding"
        )
    elif params.allele_embed_kind == "3":
        allele_embedding_ = AlleleEmbedding2(
            nalleles=params.nalleles,
            npositions=params.nmarkers,
            output_dim=params.allele_embed_dim,
            name="allele_embedding"
        )
    else:
        allele_embedding_ = layers.Embedding(
            input_dim=params.nalleles,
            output_dim=params.allele_embed_dim,
            name="allele_embedding"
        )

    allele_embedding = LayerWrapper(
        allele_embedding_,
        name="allele_embedding_model"
    )

    encoder = PerceiverEncoder(
        num_iterations=params.num_encode_iters,
        projection_units=params.projection_dim,
        ff_units=params.feedforward_dim,
        num_self_attention=params.num_self_attention,
        num_self_attention_heads=params.num_self_attention_heads,
        add_pos=True,
        share_weights=params.share_weights,
        name="encoder"
    )

    # lsize was set out of scope
    encoder(lsize)
    return (
        lsize,
        latent_initialiser,
        position_embedding,
        allele_embedding,
        encoder
    )


def build_encoder_decoder_model(
    params: Params,
):
    (
        lsize,
        latent_initialiser,
        position_embedding,
        allele_embedding,
        encoder
    ) = build_encoder_model(
        params
    )

    decoder = CrossAttention(
        projection_units=params.projection_dim,
        ff_units=params.feedforward_dim,
        name="decoder"
    )

    allele_predictor = layers.Dense(
        (params.nalleles - 1) * params.ploidy,
        activation="linear",
        name="allele_predictor"
    )

    if params.contrastive:
        contrast_predictor = keras.Sequential([
            layers.Dense(
                (params.nalleles - 1) * params.ploidy,
                activation="gelu",
            ),
            layers.Dense(
                (params.nalleles - 1),
                activation="linear",
                name="contrast_predictor"
            )
        ])
        model = TwinnedPerceiverEncoderDecoder(
            latent_initialiser=latent_initialiser,
            position_embedder=position_embedding,
            allele_embedder=allele_embedding,
            encoder=encoder,
            decoder=decoder,
            allele_predictor=(
                allele_predictor
                if params.contrastive_weight > 0
                else None
            ),
            contrast_predictor=contrast_predictor,
            relational_embedder=None,
            num_decode_iters=params.num_decode_iters,
            name="encoder_decoder",
            allele_combiner=params.allele_combiner,
            contrast_method="concat"
        )
    else:
        model = PerceiverEncoderDecoder(
            latent_initialiser=latent_initialiser,
            position_embedder=position_embedding,
            allele_embedder=allele_embedding,
            encoder=encoder,
            decoder=decoder,
            predictor=allele_predictor,
            relational_embedder=None,
            num_decode_iters=params.num_decode_iters,
            allele_combiner=params.allele_combiner,
            name="encoder_decoder"
        )
    return (
        latent_initialiser, position_embedding,
        allele_embedding, encoder, model
    )
