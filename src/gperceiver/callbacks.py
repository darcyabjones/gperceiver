#!/usr/bin/env python3

# from typing import TYPE_CHECKING

from tensorflow.keras.callbacks import Callback


class ReduceLRWithWarmup(Callback):

    def __init__(
        self,
        max_lr: float,
        warmup: int,
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

        self.warmup_counter = self.warmup
        self.cooldown_counter = 0
        self.wait = 0
        return

    def on_train_batch_end(self, batch, logs=None):
        from tensorflow.keras import backend
        logs = logs or {}

        if self.in_warmup():
            lr = float(backend.get_value(self.model.optimizer.lr))
            logs['lr'] = lr

            update = (self.max_lr - lr) / self.warmup_counter
            self.warmup_counter -= 1
            new_lr = min(lr + update, self.max_lr)
            backend.set_value(self.model.optimizer.lr, new_lr)
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

            if self.in_warmup():
                self.best = current
                self.wait = 0

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

    def in_warmup(self):
        return self.warmup_counter > 0

    def in_cooldown(self):
        return self.cooldown_counter > 0
