#!/usr/bin/env python3

import json
import sys
import traceback
import argparse
from os.path import join as pjoin

from typing import List, Optional
# import numpy.typing as npt

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

from ..preprocessing import PrepPrediction
from ..layers import (
    AlleleEmbedding,
    AlleleEmbedding2,
    PositionEmbedding,
    FourierPositionEmbedding
)

from ..models import (
    LayerWrapper,
    LatentInitialiser,
    PerceiverEncoder,
    PerceiverPredictor,
    IndexSelector
)

from ..callbacks import ReduceLRWithWarmup

from .utils import Params, build_encoder_model

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
        "phenos",
        type=argparse.FileType('r'),
        help="The phenotypes, must have columns name, response, and block"
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
        "--predictions",
        type=argparse.FileType('w'),
        help="Write the predictions to this file.",
        default=None
    )

    parser.add_argument(
        "--nalleles",
        type=int,
        help="How many different alleles are in there?",
        default=3,
    )

    parser.add_argument(
        "--ploidy",
        type=int,
        help="What's the ploidy",
        default=2,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Where to store the encoder model",
        default=None
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        help="Where the pretrained encoder to use is.",
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
        "--latent-nontrainable",
        dest="latent_trainable",
        action="store_false",
        help=(
            "Set the latent embedding to trainable during finetuning."
        ),
        default=True
    )

    parser.add_argument(
        "--encoder-nontrainable",
        dest="encoder_trainable",
        action="store_false",
        help=(
            "Set the encoder model to trainable during finetuning."
        ),
        default=True
    )

    parser.add_argument(
        "--position-embed-trainable",
        dest="position_embed_trainable",
        action="store_true",
        help=(
            "Set the positional embedding to trainable during finetuning."
        ),
        default=False
    )

    parser.add_argument(
        "--block-strategy",
        choices=[
            "pool",
            "latent",
        ],
        help="How to combine environmental data with genetic",
        default="pool"
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
        default=100
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
        "--loss",
        choices=[
            "mse",
            "mae",
            "binary",
            "poisson",
            "binary_focal"
        ],
        default="mse",
        help="What loss function should we optimise for?"
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
        "--seed",
        type=int,
        default=None,
        help="a seed"
    )

    parser.add_argument(
        "--cv",
        type=str,
        default=None,
        help="The CV entry from the 'cv' column to use for a validation set."
    )

    parsed = parser.parse_args(args)

    return parsed


def prep_dataframes(
    chroms: "pd.DataFrame",
    genos: "pd.DataFrame",
    phenos: "pd.DataFrame",
    cv: "Optional[str]" = None,
):
    assert list(chroms.columns) == ["chr", "marker", "pos"]
    chroms.sort_values(["chr", "pos"], inplace=True)
    chrom_map = np.sort(chroms["chr"].unique())
    chrom_map = dict(zip(chrom_map, range(len(chrom_map))))
    chrom_pos = chroms["chr"].apply(chrom_map.get).astype("int32").values
    marker_pos = chroms["pos"].astype("float32").values

    assert genos.columns[0] == "name"
    genos = (
        genos
        .set_index("name")
        .loc[:, chroms["marker"]]
        .astype("uint8")
    )

    assert list(phenos.columns[:3]) == ["name", "response", "block"]
    phenos["block"] = phenos["block"].astype("int32")
    phenos = phenos[(~phenos["response"].isna()) & (~phenos["block"].isna())]
    phenos.set_index("name", inplace=True)

    if ("cv" in phenos.columns) and (cv is not None):
        phenos_test = phenos[phenos["cv"].astype(str) == str(cv)]
        phenos_train = phenos[phenos["cv"].astype(str) != str(cv)]
    else:
        phenos_test = None
        phenos_train = phenos

    def inner(p, g):
        p = p[["response", "block"]]
        combined = pd.merge(
            p,
            g,
            left_index=True,
            right_index=True,
            how="inner"
        )
        p = combined[["response"]].values.astype("float64")
        b = combined[["block"]].values.astype("int32")
        g = combined[genos.columns].values.astype("int32")

        nmarkers = g.shape[1]
        nsamples = g.shape[0]
        nblocks = np.max(b) + 1
        ds = Dataset.from_tensor_slices((
            g, b, p
        ))
        return ds, nmarkers, nsamples, nblocks

    train, nmarkers, nsamples, nblocks = inner(phenos_train, genos)

    if phenos_test is not None:
        test, _, _, nblocks_test = inner(phenos_test, genos)
        nblocks = max([
            nblocks,
            nblocks_test
        ])
    else:
        test = None

    return train, test, nmarkers, nsamples, nblocks, chrom_pos, marker_pos


def prep_pred_dataframes(
    chroms: "pd.DataFrame",
    genos: "pd.DataFrame",
    phenos: "pd.DataFrame",
):
    assert list(chroms.columns) == ["chr", "marker", "pos"]
    chroms.sort_values(["chr", "pos"], inplace=True)
    chrom_map = np.sort(chroms["chr"].unique())
    chrom_map = dict(zip(chrom_map, range(len(chrom_map))))

    genos = (
        genos
        .set_index("name")
        .loc[:, chroms["marker"]]
        .astype("uint8")
    )

    del chroms
    phenos = phenos[(~phenos["response"].isna()) & (~phenos["block"].isna())]

    blocks = phenos["block"].astype("int32").unique()
    blocks = pd.DataFrame([
        {"name": n, "block": b}
        for n in np.unique(genos.index.values)
        for b in blocks
    ])
    blocks.set_index("name", inplace=True)

    combined = pd.merge(phenos, blocks, on=["name", "block"], how="outer")
    #combined = pd.merge(combined, genos, left_on="name", right_index=True)
    combined.set_index("name", inplace=True)

    ds = Dataset.from_tensor_slices(
        (genos.values.astype("int32"),)
    )
    gorder = genos.index.values

    return ds, gorder, combined


def runner(args):  # noqa
    if tf.config.list_physical_devices("GPU"):
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    chroms = pd.read_csv(args.chroms, sep="\t")
    genos = pd.read_csv(args.markers, sep="\t")
    phenos = pd.read_csv(args.phenos, sep="\t")

    (
        train,
        test,
        nmarkers,
        nsamples,
        nblocks,
        chrom_pos,
        marker_pos
    ) = prep_dataframes(
        chroms,
        genos,
        phenos,
        cv=args.cv
    )

    if args.params is not None:
        dparams = json.load(args.params)
    elif args.pretrained is not None:
        with open(pjoin(args.pretrained, "params.json"), "r") as handle:
            dparams = json.load(handle)
    else:
        dparams = {}

    params = Params.from_dict(
        dparams,
        nmarkers=nmarkers,
        nblocks=nblocks,
        chrom_pos=chrom_pos,
        marker_pos=marker_pos,
        nalleles=args.nalleles,
        ploidy=args.ploidy,
        block_strategy=args.block_strategy
    )

    if args.predictions is not None:
        predict_ds, predict_ds_index, predict_df = prep_pred_dataframes(
            chroms,
            genos,
            phenos,
        )
    else:
        predict_ds, predict_ds_index, predict_df = None, None, None

    prep = PrepPrediction(
        allele_decoder=params.allele_decoder,
        ploidy=params.ploidy,
    )

    with strategy.scope():
        (
            lsize,
            latent_initialiser,
            position_embedding,
            allele_embedding,
            encoder
        ) = build_encoder_model(params)

        if args.pretrained is not None:
            latent_initialiser = keras.models.load_model(
                pjoin(args.pretrained, "latent"),
                compile=False,
                custom_objects={"latent": LatentInitialiser}
            )
            latent_initialiser.trainable = False

            encoder = keras.models.load_model(
                pjoin(args.pretrained, "encoder"),
                compile=False,
                custom_objects={"encoder": PerceiverEncoder}
            )
            encoder.trainable = False

            if params.position_embed_kind == "fourier":
                emb_type = FourierPositionEmbedding
            else:
                emb_type = PositionEmbedding

            position_embedder = keras.models.load_model(
                pjoin(args.pretrained, "position"),
                compile=False,
                custom_objects={
                    "position_embedding": emb_type,
                    "position_embedding_model": LayerWrapper
                }
            )
            position_embedder.trainable = False

            if params.allele_embed_kind == "2":
                emb_type = AlleleEmbedding
            elif params.allele_embed_kind == "3":
                emb_type = AlleleEmbedding2
            else:
                emb_type = layers.Embedding

            allele_embedder = keras.models.load_model(
                pjoin(args.pretrained, "allele"),
                compile=False,
                custom_objects={
                    "allele_embedding": emb_type,
                    "allele_embedding_model": LayerWrapper
                }
            )
            allele_embedder.trainable = False

        if args.warmup_steps == 0:
            initial_lr = args.lr
        else:
            initial_lr = args.warmup_lr

        inner_model = PerceiverPredictor(
            nblocks=params.nblocks,
            latent_output_dim=params.output_dim,
            latent_initialiser=latent_initialiser,
            position_embedder=position_embedder,
            allele_embedder=allele_embedder,
            encoder=encoder,
            predictor=keras.Sequential([
                layers.Dense(
                    params.nblocks * 2,
                    use_bias=True,
                    activation="gelu"
                ),
                layers.Dense(1, use_bias=False),
            ]),
            intercept=layers.Embedding(
                input_dim=params.nblocks,
                output_dim=1,
                embeddings_initializer="zeros",
                name="intercept"
            ),
            allele_combiner="add",
            block_strategy=params.block_strategy,
            name="gperceiver"
        )

        model = IndexSelector(inner_model, name="selector")

        if args.checkpoint is not None:
            model.load_weights(args.checkpoint)

        if args.loss == "mse":
            loss = keras.losses.MeanSquaredError()
        elif args.loss == "mae":
            loss = keras.losses.MeanAbsoluteError()
        elif args.loss == "poisson":
            loss = keras.losses.Poisson()
        elif args.loss == "binary":
            loss = keras.losses.BinaryCrossentropy(from_logits=True)
        elif args.loss == "binary_focal":
            loss = keras.losses.BinaryFocalCrossentropy(from_logits=True)

        if args.loss in ("mse", "mae", "poisson"):
            metrics = [
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.MeanSquaredError(),
                keras.metrics.RootMeanSquaredError(),
            ]
        elif args.loss in ("binary", "binary_focal"):
            metrics = [
                # keras.metrics.Precision(thresholds=0.0),
                # keras.metrics.Recall(thresholds=0.0),
                keras.metrics.BinaryAccuracy(threshold=0.0),
            ]

        model.compile(
            optimizer=LAMB(initial_lr, weight_decay_rate=0.0001),
            loss=loss,
            metrics=metrics,
        )

    reduce_lr = ReduceLRWithWarmup(
        max_lr=args.lr,
        warmup=args.warmup_steps,
        monitor="loss",
        factor=0.1,
        patience=args.reduce_lr_patience,
        min_delta=1e-3,
        cooldown=0,
        min_lr=args.warmup_lr,
    )

    if test is None:
        monitor = "loss"
    else:
        monitor = "val_loss"

    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor,
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
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                save_freq="epoch"
            )
        )

    if args.last_checkpoint is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=args.last_checkpoint,
                monitor=monitor,
                save_best_only=False,
                save_weights_only=True,
                save_freq="epoch"
            )
        )

    batch_size = args.batch_size * strategy.num_replicas_in_sync

    train = (
        train
        .shuffle(nsamples, reshuffle_each_iteration=True)
        .apply(prep)
        .batch(batch_size)
    )

    if test is not None:
        test = (
            test
            .apply(prep)
            .batch(batch_size)
        )

    try:
        # Fit the model.
        model.fit(
            train,
            epochs=args.nepochs,
            callbacks=callbacks,
            validation_data=test,
            verbose=1
        )

        if args.pretrained is not None:
            # Only do second phase if we're finetuning.

            model.summary(expand_nested=True, show_trainable=True)

            print("Setting embedder to trainable")
            inner_model.encoder.trainable = args.encoder_trainable
            inner_model.latent_initialiser.trainable = args.latent_trainable
            inner_model.position_embedder.trainable = args.position_embed_trainable

            model.compile(
                optimizer=LAMB(initial_lr, weight_decay_rate=0.0001),
                loss=loss,
                metrics=metrics,
            )

            reduce_lr.reset()

            # Fit the model.
            model.fit(
                train,
                epochs=args.nepochs,
                callbacks=callbacks,
                validation_data=test,
                verbose=1
            )

    finally:
        model.summary(expand_nested=True, show_trainable=True)

        if args.predictions is not None:
            preds = inner_model.predict(
                predict_ds
                .apply(prep)
                .batch(batch_size)
            )
            preds = pd.DataFrame(preds, index=predict_ds_index)
            preds.index.name = "name"
            preds = pd.melt(preds, ignore_index=False, var_name="block", value_name="prediction")
            print(preds)
            preds = pd.merge(
                predict_df.reset_index(),
                preds.reset_index(),
                on=["name", "block"],
                how="outer"
            )
            preds.to_csv(args.predictions, index=False, sep="\t")

        if args.model is not None:
            inner_model.save(args.model)


def main():  # noqa
    try:
        args = cli(prog=sys.argv[0], args=sys.argv[1:])
    except MyArgumentError as e:
        print(e.message, file=sys.stderr)
        sys.exit(e.errno)

    try:
        runner(args)

    except OSError as e:
        raise e
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
