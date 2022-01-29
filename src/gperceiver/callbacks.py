#!/usr/bin/env python3

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from typing import Any, Optional
    from typing import Dict

from tensorflow.keras import layers
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

        num_iterations = int(backend.get_value(
            self.model.encoder.num_iterations
        ))
        new_num_iterations = self.schedule(epoch, num_iterations)
        backend.set_value(
            self.model.encoder.num_iterations,
            new_num_iterations
        )

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

        num_iterations = int(backend.get_value(
            self.model.num_decode_iters
        ))
        new_num_iterations = self.schedule(epoch, num_iterations)
        backend.set_value(self.model.num_decode_iters, new_num_iterations)

        if logs is not None:
            logs["ndecode_iters"] = new_num_iterations

        if self.verbose:
            print(f"Updated num decode iterations to {new_num_iterations}")
        return


class ReduceLRWithWarmup(Callback):

    def __init__(
        self,
        max_lr: float,
        warmup: int,
        pre_warmup: int = 0,
        monitor='val_loss',
        factor=0.1,
        patience=10,
        verbose=0,
        mode='auto',
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
    ):
        super(ReduceLRWithWarmup, self).__init__()
        self.max_lr = max_lr
        self.warmup = warmup
        self.pre_warmup = pre_warmup
        self.monitor = monitor

        if factor >= 1.0:
            raise ValueError("Cannot have a factor greater than 1.")

        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.wait = 0
        self.best = 0
        self.monitor_ip = None
        self.reset()
        return

    def reset(self):
        import numpy as np
        if self.mode not in ['auto', 'min', 'max']:
            self.mode = 'auto'
        if (
            self.mode == 'min' or
            (self.mode == 'auto' and 'acc' not in self.monitor)
        ):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf

        self.pre_warmup_counter = self.pre_warmup
        self.warmup_counter = self.warmup
        self.cooldown_counter = 0
        self.wait = 0
        return

    def on_epoch_end(self, epoch, logs=None):
        from tensorflow.keras import backend
        logs = logs or {}
        lr = float(backend.get_value(self.model.optimizer.lr))
        logs['lr'] = lr
        current = logs.get(self.monitor)

        if current is not None:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.in_pre_warmup():
                self.wait = 0
                self.pre_warmup_counter -= 1
            elif self.in_warmup():
                self.wait = 0
                update = (self.max_lr - lr) / self.warmup_counter
                self.warmup_counter -= 1
                new_lr = min(lr + update, self.max_lr)
                backend.set_value(self.model.optimizer.lr, new_lr)
            elif self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    new_lr = lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    backend.set_value(self.model.optimizer.lr, new_lr)
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
                    self.best = current  # S

    def in_pre_warmup(self):
        return self.pre_warmup_counter > 0

    def in_warmup(self):
        return self.warmup_counter > 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


class SetTrainableAt(Callback):

    def __init__(
        self,
        obj: "layers.Layer",
        epoch: int,
        name: str
    ):
        super(SetTrainableAt, self).__init__()
        self.obj = obj
        self.epoch = epoch
        self.name = name
        return

    def on_epoch_begin(
        self,
        epoch: int,
        logs: "Optional[Dict[Any, Any]]" = None
    ):
        logs = logs or {}
        if epoch >= self.epoch:
            self.obj.trainable = True

        logs[f"{self.name}_trainable"] = int(self.obj.trainable)
        return
