#!/usr/bin/env python3

import sys
import traceback
import argparse
import json
from os.path import join as pjoin

from typing import List

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras

from tensorflow_addons.optimizers import LAMB

from .exitcodes import (
    EXIT_VALID, EXIT_KEYBOARD, EXIT_UNKNOWN, EXIT_CLI, EXIT_INPUT_FORMAT,
    EXIT_INPUT_NOT_FOUND, EXIT_SYSERR, EXIT_CANT_OUTPUT
)

from ..preprocessing import PrepContrastData, PrepData, allele_frequencies
from ..losses import (
    PloidyBinaryCrossentropy,
    PloidyBinaryFocalCrossentropy,
)

from ..callbacks import ReduceLRWithWarmup

from .utils import Params, build_encoder_decoder_model

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
        "--model",
        type=str,
        help="Where to save the models",
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
        "--logs",
        type=str,
        help="Save a log file in tsv format",
        default=None
    )

    parser.add_argument(
        "--params",
        type=argparse.FileType('r'),
        help="The model hyperparameters",
        default=None
    )

    parser.add_argument(
        "--nalleles",
        type=int,
        help="How many different alleles are in there?",
        default=None,
    )

    parser.add_argument(
        "--ploidy",
        type=int,
        help="What's the ploidy",
        default=None,
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="The maximum learning rate",
        default=1e-3
    )

    parser.add_argument(
        "--warmup-lr",
        type=float,
        help="The learning rate to start from during warmup",
        default=1e-10
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        help="The number of steps (minibatches) to ramp up the learning rate",
        default=100,
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
        "--prop-x",
        type=float,
        help=(
            "The proportion of markers to use as features."
        ),
        default=1.0
    )

    parser.add_argument(
        "--prop-y",
        type=float,
        help=(
            "The proportion of markers to predict."
        ),
        default=0.1
    )

    parser.add_argument(
        "--contrastive",
        action="store_true",
        help="Learn by contrasting samples",
        default=False,
    )

    parser.add_argument(
        "--contrastive-allele-weight",
        dest="contrastive_weight",
        type=float,
        default=0.0,
        help=(
            "Weight allele weight for contrastive loss by this amount. "
            "Should be <= 1. "
            "0 disables allele weight completely, so only contrastive weight. "
            "1 means that allele and contrast loss contribute equally."
        )
    )

    parser.add_argument(
        "--ploidy-scaler",
        type=float,
        default=1.0,
        help="Weight ploidy penalty by this amount."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="a seed"
    )

    parser.add_argument(
        "--loss",
        choices=[
            "binary",
            "binary_focal"
        ],
        default="binary_focal",
        help="What loss function should we optimise for?"
    )

    parsed = parser.parse_args(args)

    return parsed


def runner(args):  # noqa
    assert 0 < args.prop_x <= 1
    assert 0 < args.prop_y < 1

    if tf.config.list_physical_devices("GPU"):
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    chroms = pd.read_csv(args.chroms, sep="\t")
    assert list(chroms.columns) == ["chr", "marker", "pos"]
    chroms.sort_values(["chr", "pos"], inplace=True)
    chrom_map = np.sort(chroms["chr"].unique())
    chrom_map = dict(zip(chrom_map, range(len(chrom_map))))
    chrom_pos = chroms["chr"].apply(chrom_map.get).astype("int32").values
    marker_pos = chroms["pos"].astype("float32").values

    genos = pd.read_csv(args.markers, sep="\t")
    assert genos.columns[0] == "name"
    genos.set_index("name", inplace=True)
    genos = genos.loc[:, chroms["marker"]].astype("uint8")

    nmarkers = genos.shape[1]
    nsamples = genos.shape[0]

    train = Dataset.from_tensor_slices(genos.values.astype("int32"))

    if args.params is not None:
        dparams = json.load(args.params)
    else:
        dparams = {}

    params = Params.from_dict(
        dparams,
        nmarkers=nmarkers,
        chrom_pos=chrom_pos,
        marker_pos=marker_pos,
        nalleles=args.nalleles,
        ploidy=args.ploidy,
        contrastive=args.contrastive,
        contrastive_weight=args.contrastive_weight
    )

    with tf.device("cpu:0"):
        freqs = allele_frequencies(
            tf.gather(params.allele_decoder, genos.values.astype("int32")),
            params.nalleles
        )
    del genos

    if params.contrastive:
        prep_aec = PrepContrastData(
            allele_frequencies=freqs,
            allele_decoder=params.allele_decoder,
            ploidy=params.ploidy,
            prop_x=args.prop_x,
            prop_y=args.prop_y,
            seed=args.seed,
        )
    else:
        prep_aec = PrepData(
            allele_frequencies=freqs,
            allele_decoder=params.allele_decoder,
            ploidy=params.ploidy,
            prop_x=args.prop_x,
            prop_y=args.prop_y,
            seed=args.seed,
        )

    with strategy.scope():
        (
            latent_initialiser,
            position_embedding,
            allele_embedding,
            encoder,
            model,
        ) = build_encoder_decoder_model(params)

        if args.warmup_steps == 0:
            initial_lr = args.lr
        else:
            initial_lr = args.warmup_lr

        if args.loss == "binary":
            allele_pred_loss = PloidyBinaryCrossentropy(
                from_logits=True,
                ploidy=params.ploidy,
                ploidy_scaler=args.ploidy_scaler,
                name="allele_crossentropy"
            )
            contrast_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True, label_smoothing=0.0, axis=-1,
                name='binary_crossentropy'
            )
        elif args.loss == "binary_focal":
            allele_pred_loss = PloidyBinaryFocalCrossentropy(
                from_logits=True,
                ploidy=params.ploidy,
                ploidy_scaler=args.ploidy_scaler,
                name="allele_crossentropy"
            )

            contrast_loss = tf.keras.losses.BinaryFocalCrossentropy(
                from_logits=True, label_smoothing=0.0, axis=-1,
                name='binary_crossentropy'
            )
        else:
            raise ValueError(f"Invalid loss {args.loss}")

        if params.contrastive and (params.contrastive_weight > 0):
            loss_fns = [
                contrast_loss,
                allele_pred_loss,
                allele_pred_loss
            ]
        else:
            loss_fns = allele_pred_loss

        metrics = keras.metrics.BinaryAccuracy(
            threshold=0.0,
            name="accuracy"
        )

        model.compile(
            optimizer=LAMB(initial_lr, weight_decay_rate=0.0001),
            loss=loss_fns,
            metrics=metrics,
        )

    if args.checkpoint is not None:
        model.load_weights(args.checkpoint)

    # position_embedding.position_embedder.trainable = args.position_embed_trainable  # noqa

    reduce_lr = ReduceLRWithWarmup(
        max_lr=args.lr,
        warmup=args.warmup_steps,
        monitor='loss',
        factor=0.1,
        patience=args.reduce_lr_patience,
        min_delta=1e-3,
        cooldown=0,
        min_lr=args.warmup_lr,
    )

    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=args.early_stopping_patience,
        restore_best_weights=False
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

    dataset = (
        train
        .shuffle(nsamples + 1, reshuffle_each_iteration=True)
        .apply(prep_aec)
    )
    if params.contrastive:
        contrast_weight = 0.5 * params.contrastive_weight
        dataset = dataset.map(
            lambda x, y: (x, y, (1, contrast_weight, contrast_weight))
        )

    try:
        # Fit the model.
        model.fit(
            dataset.batch(batch_size),
            epochs=args.nepochs,
            callbacks=callbacks,
            verbose=0
        )
    finally:
        model.summary(expand_nested=True, show_trainable=True)

        if args.model is not None:
            model.save(pjoin(args.model, "encoder_decoder"))
            model.encoder.save(pjoin(args.model, "encoder"))
            model.latent_initialiser.save(pjoin(args.model, "latent"))
            model.position_embedder.save(pjoin(args.model, "position"))
            model.allele_embedder.save(pjoin(args.model, "allele"))
            with open(pjoin(args.model, "params.json"), "w") as handle:
                json.dump(params.to_dict(), handle, indent=2)


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
