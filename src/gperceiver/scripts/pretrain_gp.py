#!/usr/bin/env python3

import sys
import traceback
import argparse

from typing import List, Callable

import pandas as pd

from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow_addons.optimizers import LAMB

from .exitcodes import (
    EXIT_VALID, EXIT_KEYBOARD, EXIT_UNKNOWN, EXIT_CLI, EXIT_INPUT_FORMAT,
    EXIT_INPUT_NOT_FOUND, EXIT_SYSERR, EXIT_CANT_OUTPUT
)

from ..preprocessing import pairwise_correlation, prep_aec
from ..layers import (
    PrepMarkers,
    PerceiverBlock,
    PerceiverDecoderBlock,
    SquareRelu,
)

from ..models import (
    PerceiverLatentInitialiser,
    PerceiverMarkerEncoder,
    PerceiverEncoderDecoder,
)

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
        "--encoder",
        type=str,
        help="Where to store the encoder model",
        default="encoder"
    )

    parser.add_argument(
        "--latent",
        type=str,
        help="Where to store the latent model",
        default="latent"
    )

    parser.add_argument(
        "--encoder-decoder",
        type=str,
        help="Where to store the combined encoder/decoder model",
        default="encoder_decoder"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Start from this model instead of a new one",
        default=None
    )

    parser.add_argument(
        "--marker-embed-dim",
        type=int,
        help="The number of dimensions for the per-marker learned embeddings",
        default=4
    )

    parser.add_argument(
        "--projection-dim",
        type=int,
        help="The number of channels to use for attention comparisons",
        default=64
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        help="The number of learned latent features to use",
        default=64
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
        default=2
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
        help="The number iterations to run through the decoder network",
        default=4
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="The initial learning rate",
        default=1e-3
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
    parsed = parser.parse_args(args)

    return parsed


def make_epoch_scheduler(
    max_iters: int,
    frequency: int = 1
) -> "Callable[[int, int], int]":
    def scheduler(epoch: int, num_iterations: int) -> int:
        if (
            (epoch > 0)
            and (epoch % frequency == 0)
            and (num_iterations < max_iters)
        ):
            num_iterations += 1
        return num_iterations

    return scheduler


def runner(args):
    genos = pd.read_csv(args.markers, sep="\t")
    genos.set_index("name", inplace=True)

    train = Dataset.from_tensor_slices(genos.values)

    if args.checkpoint is None:
        hamming_positions = pairwise_correlation(genos.values)
        latent_initialiser = PerceiverLatentInitialiser(
            args.output_dim,
            args.latent_dim,
            name="latent"
        )
        encoder = PerceiverMarkerEncoder(
            PrepMarkers(
                positions=hamming_positions,
                embed_dim=args.marker_embed_dim,
                output_dim=args.projection_dim,
                name="prep_markers"
            ),
            perceivers=[
                PerceiverBlock(
                    projection_units=args.projection_dim,
                    num_heads=args.num_sa_heads,
                    num_self_attention=args.num_sa,
                    name="perceiver_block"
                )
            ],
            num_iterations=args.num_encode_iters,
            name="encoder"
        )

        lsize = [
            layers.Input(train.map(prep_aec).element_spec[0].shape),
            layers.Input((None, args.output_dim))
        ]
        encoder(lsize)
        model1 = PerceiverEncoderDecoder(
            latent_initialiser=latent_initialiser,
            encoder=encoder,
            decoder=PerceiverDecoderBlock(
                projection_units=args.projection_dim,
                name="decoder"
            ),
            predictor=keras.Sequential([
                layers.Dropout(0.5),
                layers.Dense(32, activation=SquareRelu()),
                layers.Dropout(0.5),
                layers.Dense(3, activation="softmax")]
            ),
            num_decode_iters=args.num_decode_iters,
            name="encoder_decoder"
        )
    else:
        model1 = keras.models.load_model(
            args.checkpoint,
            custom_objects={
                "encoder_decoder": PerceiverEncoderDecoder,
                "decoder": PerceiverDecoderBlock,
                "encoder": PerceiverMarkerEncoder,
                "prep_markers": PrepMarkers,
                "perceiver_block": PerceiverBlock,
                "latent": PerceiverLatentInitialiser
            },
            compile=False
        )

    model1.compile(
        optimizer=LAMB(args.lr, weight_decay_rate=0.0001),
        loss="categorical_crossentropy",
        metrics="accuracy",
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.1, patience=5
    )

    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True
    )

    # Fit the model.
    try:
        model1.fit(
            train.shuffle(genos.shape[0]).map(prep_aec).batch(args.batch_size),
            epochs=args.nepochs,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
    except Exception as e:
        raise e
    finally:
        model1.save(args.encoder_decoder)
        model1.encoder.save(args.encoder)
        model1.latent_initialiser.save(args.latent)
    return


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
