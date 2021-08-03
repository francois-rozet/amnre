#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from itertools import islice
from time import time
from typing import Tuple


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


class Dummy(nn.Module):
    def __getitem__(self, idx):
        return self

    def forward(self, *args, **kwargs):
        return None

    def embedding(self, *args, **kwargs):
        return None


def routine(
    model: nn.Module,
    dataloader: data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer = None,
    adversary: nn.Module = None,
    inverse: bool = False,
    descents: int = None,
    flow: bool = False,
    mask_sampler: nn.Module = None,
    clip: float = 1e2,
) -> Tuple[float, torch.Tensor]:  # (time, losses)
    r"""Training routine"""

    if adversary is None:
        adversary = Dummy()

    losses = []

    start = time()

    for theta, theta_prime, x in islice(dataloader, descents):
        y = model.embedding(x)
        adv_y = adversary.embedding(x)

        if flow:
            prob = model(theta, y)
            l = criterion(prob)
        else:
            if mask_sampler is None:
                ratio = model(theta, y)
                ratio_prime = model(theta_prime, y)
                adv_ratio = adversary(theta if inverse else theta_prime, adv_y)
            else:
                if model.hyper is None:
                    mask = mask_sampler(theta.shape[:1])
                else:
                    mask = mask_sampler()

                ratio = model(theta, y, mask)
                ratio_prime = model(theta_prime, y, mask)
                adv_ratio = adversary(theta if inverse else theta_prime, adv_y, mask)

            if adv_ratio is not None:
                adv_ratio = (-adv_ratio if inverse else adv_ratio).exp()

            if inverse:
                l = criterion(ratio, adv_ratio) + criterion(-ratio_prime)
            else:
                l = criterion(ratio) + criterion(-ratio_prime, adv_ratio)

        if not l.isfinite():
            continue

        losses.append(l.item())

        if optimizer is not None:
            optimizer.zero_grad()
            l.backward()

            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

    end = time()

    return end - start, torch.tensor(losses)
