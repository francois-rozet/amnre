#!/usr/bin/env python

import random
import torch
import torch.utils.data as data

from typing import Tuple

from simulators import Simulator


class LTEDataset(data.Dataset):
    r"""Likelihood-To-Evidence Dataset"""

    def __init__(self, sim: Simulator):
        super().__init__()

        self.sim = sim
        self.theta, self.x = [], []

        self._true = torch.ones(self.sim.theta_size, dtype=bool)
        self._false = ~self._true

    def save(self, f: str) -> None:
        torch.save((self.theta, self.x), f)

    def load(self, f: str):  # -> self
        self.theta, self.x = torch.load(f)
        return self

    def simulate(self, size: int, batch: int = 1024):  # -> self
        theta, x = [], []

        for s in [batch] * (size // batch) + [size % batch]:
            samples = self.sim.sample((s,))

            theta.append(samples[0].cpu())
            x.append(samples[1].cpu())

        self.theta, self.x = torch.cat(theta), torch.cat(x)
        return self

    def __len__(self) -> int:
        return len(self.theta)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        theta, x = self.theta[i], self.x[i]

        mask = self.random_mask()
        theta = mask * theta + ~mask * self.sim.prior.sample()

        return theta, x, mask

    def random_mask(self) -> torch.BoolTensor:
        if random.random() < 0.5:
            return self._true
        else:
            return self._false


class CLTEDataset(LTEDataset):
    r"""Conditional Likelihood-To-Evidence Dataset"""

    def __init__(self, masks: torch.BoolTensor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.masks = masks

    def random_mask(self) -> torch.Tensor:
        if random.random() < 0.5:
            return self._true
        else:
            return ~random.choice(self.masks)


class MLTEDataset(LTEDataset):
    r"""Marginal Likelihood-To-Evidence Dataset"""

    def __init__(self, masks: torch.BoolTensor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.masks = masks

    def random_mask(self) -> torch.Tensor:
        if random.random() < 0.5:
            return random.choice(self.masks)
        else:
            return self._false
