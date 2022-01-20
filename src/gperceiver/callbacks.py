#!/usr/bin/env python3

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from typing import Any, Optional
    from typing import Dict

from tensorflow.keras import backend
from tensorflow.keras.callbacks import Callback


class IncreaseEncodeIterations(Callback):

    def __init__(
        self,
        schedule: "Callable[[int, int], int]",
        verbose: bool = False
    ):
        super(IncreaseEncodeIterations, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        return

    def on_epoch_begin(
        self,
        epoch: int,
        logs: "Optional[Dict[Any, Any]]" = None
    ):
        if not hasattr(self.model.encoder, "num_iterations"):
            raise ValueError("The encoder needs to have num_iterations set")

        num_iterations = float(backend.get_value(
            self.model.encoder.num_iterations
        ))
        new_num_iterations = self.schedule(epoch, num_iterations)
        backend.set_value(self.model.encoder.num_iterations, new_num_iterations)

        if logs is not None:
            logs["nencode_iters"] = new_num_iterations

        if self.verbose:
            print(f"Updated num encode iterations to {new_num_iterations}")
        return


class IncreaseDecodeIterations(Callback):

    def __init__(
        self,
        schedule: "Callable[[int, int], int]",
        verbose: bool = False
    ):
        super(IncreaseDecodeIterations, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        return

    def on_epoch_begin(
        self,
        epoch: int,
        logs: "Optional[Dict[Any, Any]]" = None
    ):
        if not hasattr(self.model, "num_decode_iters"):
            raise ValueError("The encoder needs to have num_iterations set")

        num_iterations = float(backend.get_value(
            self.model.num_decode_iters
        ))
        new_num_iterations = self.schedule(epoch, num_iterations)
        backend.set_value(self.model.num_decode_iters, new_num_iterations)

        if logs is not None:
            logs["ndecode_iters"] = new_num_iterations

        if self.verbose:
            print(f"Updated num decode iterations to {new_num_iterations}")
        return
