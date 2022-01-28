#!/usr/bin/env python3

import sys
import traceback
import argparse

from typing import List

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow_addons.optimizers import LAMB

from .exitcodes import (
    EXIT_VALID, EXIT_KEYBOARD, EXIT_UNKNOWN, EXIT_CLI, EXIT_INPUT_FORMAT,
    EXIT_INPUT_NOT_FOUND, EXIT_SYSERR, EXIT_CANT_OUTPUT
)

from ..preprocessing import pairwise_correlation, PrepAEC
from ..layers import (
    CrossAttention,
)

from ..models import (
    LatentInitialiser,
    MarkerEmbedding,
    PerceiverEncoder,
    PerceiverEncoderDecoder,
)

from ..initializers import FourierEncoding, InitializeWithValues

from ..callbacks import ReduceLRWithWarmup, SetTrainableAt

__email__ = "darcy.ab.jones@gmail.com"


class MyArgumentParser(argparse.ArgumentParser):

    def error(self, message: str):
        """ Override default to have more informative exit codes. """
        self.print_usage(sys.stderr)
        raise MyArgumentError("{}: error: {}".format(self.prog, message))


class MyArgumentError(Exception):

    def __init__(self, message: str):
        self.message = message
        self.errno = EXIT_CLI

        # This is a bit hacky, but I can't figure out another way to do it.
        if "No such file or directory" in message:
            if "infile" in message:
                self.errno = EXIT_INPUT_NOT_FOUND
            elif "outfile" in message:
                self.errno = EXIT_CANT_OUTPUT
        return


def cli(prog: str, args: List[str]) -> argparse.Namespace:
    parser = MyArgumentParser(
        prog=prog,
        description=(
            "Examples:"
        ),
        epilog=(
            "Exit codes:\n\n"
            f"{EXIT_VALID} - Everything's fine\n"
            f"{EXIT_KEYBOARD} - Keyboard interrupt\n"
            f"{EXIT_CLI} - Invalid command line usage\n"
            f"{EXIT_INPUT_FORMAT} - Input format error\n"
            f"{EXIT_INPUT_NOT_FOUND} - Cannot open the input\n"
            f"{EXIT_SYSERR} - System error\n"
            f"{EXIT_CANT_OUTPUT} - Can't create output file\n"
            f"{EXIT_UNKNOWN} - Unhandled exception, please file a bug!\n"
        )
    )

    parser.add_argument(
        "markers",
        type=argparse.FileType('r'),
        help="The markers"
    )

    parser.add_argument(
        "--nalleles",
        type=int,
        help="How many different alleles are in there?",
        default=3,
    )

    parser.add_argument(
        "--encoder",
        type=str,
        help="Where to store the encoder model",
        default=None
    )

    parser.add_argument(
        "--latent",
        type=str,
        help="Where to store the latent model",
        default=None
    )

    parser.add_argument(
        "--encoder-decoder",
        type=str,
        help="Where to store the combined encoder/decoder model",
        default=None
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Start from this model instead of a new one",
        default=None
    )

    parser.add_argument(
        "--best-checkpoint",
        type=str,
        help="Save the best checkpoint here",
        default=None
    )

    parser.add_argument(
        "--last-checkpoint",
        type=str,
        help="Start checkpoint from most recent epoch here.",
        default=None
    )

    parser.add_argument(
        "--marker-embed-dim",
        type=int,
        help="The number of dimensions for the per-marker learned embeddings",
        default=128
    )

    parser.add_argument(
        "--relational-embed",
        choices=["tsvd", "normal", "none"],
        help="Add a relational embedding to the marker positions",
        default="none"
    )

    parser.add_argument(
        "--relational-embed-trainable",
        type=int,
        help=(
            "How many epochs to wait before setting the "
            "relational embeddings to be trainable. "
            "0 lets train from beginning, -1 (default)"
            "means it wont ever be trainable."
        ),
        default=-1
    )

    parser.add_argument(
        "--position-embed-trainable",
        type=int,
        help=(
            "How many epochs to wait before setting the "
            "positional encodings to be trainable. "
            "0 lets train from beginning, -1 (default)"
            "means it wont ever be trainable."
        ),
        default=-1
    )

    parser.add_argument(
        "--projection-dim",
        type=int,
        help="The number of channels to use for attention comparisons",
        default=128
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        help="The number of learned latent features to use",
        default=128
    )

    parser.add_argument(
        "--output-dim",
        type=int,
        help="The number of channels to use in latent features",
        default=128
    )

    parser.add_argument(
        "--num-sa-heads",
        type=int,
        help="The number heads to use for self attention",
        default=4
    )

    parser.add_argument(
        "--num-sa",
        type=int,
        help="The number of transformers to use for self attention",
        default=2
    )

    parser.add_argument(
        "--num-encode-iters",
        type=int,
        help="The number iterations to run through the encoder network",
        default=4
    )

    parser.add_argument(
        "--num-decode-iters",
        type=int,
        help="The number of iterations to run through the decoder network",
        default=2
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="The maximum learning rate",
        default=1e-3
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        help="The number of epochs to ramp up the learning rate",
        default=9,
    )

    parser.add_argument(
        "--pre-warmup-epochs",
        type=int,
        help=(
            "The number of epochs run with a minimal learning rate "
            "before beginning the warmup."
        ),
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="The batch size",
        default=32
    )

    parser.add_argument(
        "--nepochs",
        type=int,
        help="The maximum number of epochs",
        default=500
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help=(
            "How many epochs since we've seen an "
            "improvement should we run?"
        ),
        default=50
    )

    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        help=(
            "How many epochs since we've seen an improvement "
            "should we run before reducing the lr?"
        ),
        default=10
    )

    parser.add_argument(
        "--prop-dropped",
        type=float,
        help=(
            "The proportion markers to drop out for prediction."
        ),
        default=0.1
    )

    parser.add_argument(
        "--use-linkage-positions",
        action="store_true",
        help="Use hamming distance for positional encodings.",
        default=False
    )

    parser.add_argument(
        "--logs",
        type=str,
        help="Save a log file in tsv format",
        default=None
    )

    parser.add_argument(
        "--encoder-nontrainable",
        action="store_true",
        help="set the encoder to be non-trainable",
        default=False,
    )

    parser.add_argument(
        "--decoder-nontrainable",
        action="store_true",
        help="set the decoder to be non-trainable",
        default=False,
    )

    parser.add_argument(
        "--latent-nontrainable",
        action="store_true",
        help="set the latent to be non-trainable",
        default=False,
    )

    parser.add_argument(
        "--marker-nontrainable",
        action="store_true",
        help="set the marker to be non-trainable",
        default=False,
    )

    parser.add_argument(
        "--predictor-nontrainable",
        action="store_true",
        help="set the predictor to be non-trainable",
        default=False,
    )

    parsed = parser.parse_args(args)

    return parsed


def runner(args):  # noqa
    assert 0 < args.prop_dropped < 1

    if tf.config.list_physical_devices("GPU"):
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    genos = pd.read_csv(args.markers, sep="\t")
    genos.set_index("name", inplace=True)

    nmarkers = genos.shape[1]
    nsamples = genos.shape[0]

    if args.relational_embed == "tsvd":
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import RobustScaler

        tsvd = Pipeline([
            ("tsvd", TruncatedSVD(n_components=args.projection_dim)),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0)))
        ])
        tsvd.fit(genos.T)
        pca = tsvd.transform(genos.T)
        del tsvd
    else:
        pca = None

    if args.use_linkage_positions:
        positions = np.cumsum(pairwise_correlation(genos.values))
    else:
        positions = None

    train = Dataset.from_tensor_slices(genos.values)
    prep_aec = PrepAEC(offset=1, prop_ones=1 - args.prop_dropped)

    del genos

    # This stuff initialises it so that the encoder can take
    # variable length sequences
    lsize = [
        layers.Input((None, args.output_dim)),
        layers.Input((None, args.marker_embed_dim)),
    ]

    if args.relational_embed != "none":
        lsize.append(
            layers.Input((None, args.projection_dim))
        )

    with strategy.scope():
        latent_initialiser = LatentInitialiser(
            args.output_dim,
            args.latent_dim,
            name="latent"
        )

        marker_embedding = MarkerEmbedding(
            nalleles=args.nalleles + 1,
            npositions=nmarkers,
            output_dim=args.marker_embed_dim,
            position_embeddings_initializer=FourierEncoding(
                positions=positions
            ),
            position_embeddings_trainable=args.position_embed_trainable == 0,  # noqa
            name="prep_markers"
        )

        if args.relational_embed == "rsvd":
            rel_embed = layers.Embedding(
                input_dim=nmarkers,
                output_dim=args.projection_dim,
                embeddings_initializer=InitializeWithValues(values=pca),
                trainable=args.relational_embed_trainable == 0,
                name="relational_embedder"
            )
            del pca
        elif args.relational_embed == "normal":
            rel_embed = layers.Embedding(
                input_dim=nmarkers,
                output_dim=args.projection_dim,
                embeddings_initializer="random_normal",
                trainable=True,
                name="relational_embedder"
            )
        else:
            rel_embed = None

        encoder = PerceiverEncoder(
            num_iterations=args.num_encode_iters,
            projection_units=args.projection_dim,
            num_self_attention=args.num_sa,
            num_self_attention_heads=args.num_sa_heads,
            add_pos=True,
            share_weights="after_first_xa",
            name="encoder"
        )

        # lsize was set out of scope
        encoder(lsize)

        decoder = CrossAttention(
            projection_units=args.projection_dim,
            name="decoder"
        )
        predictor = layers.Dense(
            args.nalleles,
            activation="softmax",
            name="predictor"
        )

        model1 = PerceiverEncoderDecoder(
            latent_initialiser=latent_initialiser,
            marker_embedder=marker_embedding,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            relational_embedder=rel_embed,
            num_decode_iters=args.num_decode_iters,
            name="encoder_decoder"
        )

        if (args.warmup_epochs == 0) and (args.pre_warmup_epochs == 0):
            initial_lr = args.lr
        else:
            initial_lr = 1e-10

        model1.compile(
            optimizer=LAMB(initial_lr, weight_decay_rate=0.0001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics="accuracy",
        )

    if args.checkpoint is not None:
        model1.load_weights(args.checkpoint)

    if args.encoder_nontrainable:
        model1.encoder.trainable = False

    if args.decoder_nontrainable:
        model1.decoder.trainable = False

    if args.latent_nontrainable:
        model1.latent_initialiser.trainable = False

    if args.marker_nontrainable:
        model1.marker_embedder.trainable = False

    if args.predictor_nontrainable:
        model1.predictor.trainable = False

    if (args.relational_embed != "none") and (args.relational_embed_trainable == 0):
        rel_embed.trainable = True

    if args.position_embed_trainable == 0:
        marker_embedding.position_embedder.trainable = True

    reduce_lr = ReduceLRWithWarmup(
        max_lr=args.lr,
        pre_warmup=args.pre_warmup_epochs,
        warmup=args.warmup_epochs,
        monitor='loss',
        factor=0.1,
        patience=args.reduce_lr_patience,
        min_delta=1e-3,
        cooldown=0,
        min_lr=1e-10,
    )

    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=args.early_stopping_patience,
        restore_best_weights=True
    )

    callbacks = [early_stopping, reduce_lr]

    if (
        (args.relational_embed == "tsvd")
        and (args.relational_embed_trainable > 0)
    ):
        callbacks.append(
            SetTrainableAt(
                rel_embed,
                args.relational_embed_trainable,
                "relational_embedder"
            )
        )

    if args.position_embed_trainable > 0:
        callbacks.append(
            SetTrainableAt(
                marker_embedding.position_embedder,
                args.position_embed_trainable,
                "position_embedder"
            )
        )

    if args.logs is not None:
        callbacks.append(
            keras.callbacks.CSVLogger(args.logs, separator="\t", append=True)
        )

    if args.best_checkpoint is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=args.best_checkpoint,
                monitor="loss",
                save_best_only=True,
                save_weights_only=True,
                save_freq="epoch"
            )
        )

    if args.last_checkpoint is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=args.last_checkpoint,
                monitor="loss",
                save_best_only=False,
                save_weights_only=True,
                save_freq="epoch"
            )
        )

    batch_size = args.batch_size * strategy.num_replicas_in_sync

    # Fit the model.
    model1.fit(
        train.shuffle(nsamples).map(prep_aec).batch(batch_size),
        epochs=args.nepochs + args.warmup_epochs + args.pre_warmup_epochs,
        callbacks=callbacks,
        verbose=1
    )

    if args.encoder_decoder is not None:
        model1.save(args.encoder_decoder)

    if args.encoder is not None:
        model1.encoder.save(args.encoder)

    if args.latent is not None:
        model1.latent_initialiser.save(args.latent)


def main():  # noqa
    try:
        args = cli(prog=sys.argv[0], args=sys.argv[1:])
    except MyArgumentError as e:
        print(e.message, file=sys.stderr)
        sys.exit(e.errno)

    try:
        runner(args)

    except OSError as e:
        msg = (
            "Encountered a system error.\n"
            "We can't control these, and they're usually related to your OS.\n"
            "Try running again.\n"
        )
        print(msg, file=sys.stderr)
        print(e.strerror, file=sys.stderr)
        sys.exit(EXIT_SYSERR)

    except MemoryError:
        msg = (
            "Ran out of memory!\n"
            "Catastrophy shouldn't use much RAM, so check other "
            "processes and try running again."
        )
        print(msg, file=sys.stderr)
        sys.exit(EXIT_SYSERR)

    except KeyboardInterrupt:
        print("Received keyboard interrupt. Exiting.", file=sys.stderr)
        sys.exit(EXIT_KEYBOARD)

    except Exception as e:
        msg = (
            "I'm so sorry, but we've encountered an unexpected error.\n"
            "This shouldn't happen, so please file a bug report with the "
            "authors.\nWe will be extremely grateful!\n\n"
            "You can email us at {}.\n"
            "Alternatively, you can file the issue directly on the repo "
            "<https://github.com/darcyabjones/selectml/issues>\n\n"
            "Please attach a copy of the following message:"
        ).format(__email__)
        print(e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(EXIT_UNKNOWN)

    return


if __name__ == '__main__':
    main()
