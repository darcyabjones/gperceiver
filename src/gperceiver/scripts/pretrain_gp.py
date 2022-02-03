#!/usr/bin/env python3

import sys
import traceback
import argparse

from typing import List

import pandas as pd

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from tensorflow_addons.optimizers import LAMB

from .exitcodes import (
    EXIT_VALID, EXIT_KEYBOARD, EXIT_UNKNOWN, EXIT_CLI, EXIT_INPUT_FORMAT,
    EXIT_INPUT_NOT_FOUND, EXIT_SYSERR, EXIT_CANT_OUTPUT
)

from ..preprocessing import PrepAEC
from ..layers import (
    CrossAttention,
)

from ..models import (
    LatentInitialiser,
    PositionEmbedding,
    AlleleEmbedding,
    PerceiverEncoder,
    PerceiverEncoderDecoder,
)

from ..callbacks import ReduceLRWithWarmup

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
        "chroms",
        type=argparse.FileType('r'),
        help="The chromosomes and positions of the markers"
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
        default=4
    )

    parser.add_argument(
        "--position-embed-dim",
        type=int,
        help="The number of dimensions for the position embeddings",
        default=256
    )

    parser.add_argument(
        "--position-embed-trainable",
        action="store_true",
        help=(
            "How many epochs to wait before setting the "
            "positional encodings to be trainable. "
            "0 lets train from beginning, -1 (default)"
            "means it wont ever be trainable."
        ),
        default=False
    )

    parser.add_argument(
        "--share-weights",
        choices=["after_first_xa", "true", "false", "after_first"],
        default="after_first_xa",
        help="Should we share weights between layers"
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
        default=256
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
        default=4,
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
        default=16
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

    chroms = pd.read_csv(args.chroms, sep="\t")
    assert list(genos.columns) == ["chrom", "marker", "pos"]

    genos = pd.read_csv(args.markers, sep="\t")
    assert genos.columns[0] == "name"
    genos.set_index("name", inplace=True)

    nchroms = genos["chrom"].nunique()
    chroms = genos["chrom"].values
    positions = tf.convert_to_tensor(genos["pos"].values, dtype=tf.float32)

    genos = genos.iloc[:, 2:].values.T

    nmarkers = genos.shape[1]
    nsamples = genos.shape[0]

    train = Dataset.from_tensor_slices(genos.values)
    prep_aec = PrepAEC(offset=1, prop_ones=1 - args.prop_dropped)

    del genos

    # This stuff initialises it so that the encoder can take
    # variable length sequences
    lsize = [
        layers.Input((None, args.output_dim)),
        layers.Input((None, args.marker_embed_dim + args.position_embed_dim)),
    ]

    if args.share_weights == "true":
        args.share_weights = True

    elif args.share_weights == "false":
        args.share_weights = False

    with strategy.scope():
        latent_initialiser = LatentInitialiser(
            args.output_dim,
            args.latent_dim,
            name="latent"
        )

        position_embedding = PositionEmbedding(
            positions=positions,
            nchroms=nchroms if nchroms > 1 else None,
            output_dim=args.position_embed_dim,
            position_embeddings_trainable=args.position_embed_trainable,
            name="position_embedding",
        )

        allele_embedding = AlleleEmbedding(
            nalleles=args.nalleles + 1,
            npositions=nmarkers,
            output_dim=args.marker_embed_dim,
            embeddings_regularizer=regularizers.L2(1e-10),
            name="allele_embedding"
        )

        encoder = PerceiverEncoder(
            num_iterations=args.num_encode_iters,
            projection_units=args.projection_dim,
            num_self_attention=args.num_sa,
            num_self_attention_heads=args.num_sa_heads,
            add_pos=True,
            share_weights=args.share_weights,
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
            position_embedder=position_embedding,
            allele_embedder=allele_embedding,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            relational_embedder=None,
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

    position_embedding.position_embedder.trainable = args.position_embed_trainable  # noqa

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
        train.shuffle(nsamples).batch(batch_size).map(prep_aec),
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
            "Please attach a copy of the following message:"
        ).format(__email__)
        print(e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(EXIT_UNKNOWN)

    return


if __name__ == '__main__':
    main()
