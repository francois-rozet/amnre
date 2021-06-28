#!/usr/bin/env python

import torch.optim as optim

class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, *args, **kwargs):
        self.plateau = False
        return super().step(*args, **kwargs)

    def _reduce_lr(self, *args, **kwargs):
        self.plateau = True
        return super()._reduce_lr(*args, **kwargs)

    @property
    def lr(self) -> float:
        return self._last_lr[0]

    @property
    def bottom(self) -> bool:
        return self.lr <= self.min_lrs[0]
