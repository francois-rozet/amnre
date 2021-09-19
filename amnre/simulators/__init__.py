#!/usr/bin/env python

import math
import numpy as np
import os
import psutil
import random
import torch
import torch.nn as nn
import torch.utils.data as data

from functools import cached_property
from torch.distributions import Distribution
from typing import Any, Tuple, List


Distribution.set_default_validate_args(False)


class Simulator(nn.Module):
    r"""Abstract Simulator"""

    @cached_property
    def prior(self) -> Distribution:
        r""" p(theta) """

        return self.masked_prior(...)

    def masked_prior(self, mask: torch.BoolTensor) -> Distribution:
        r""" p(theta_a) """

        raise NotImplementedError()

    @cached_property
    def labels(self) -> List[str]:  # parameters' labels
        theta_size = self.prior.sample().numel()
        labels = [f'$\\theta_{{{i}}}$' for i in range(1, theta_size + 1)]

        return labels

    def likelihood(self, theta: torch.Tensor) -> Distribution:
        r""" p(x | theta) """

        raise NotImplementedError()

    @cached_property
    def tractable(self) -> bool:
        theta = self.prior.sample()

        try:
            lkh = self.likelihood(theta)
            return True
        except NotImplementedError as e:
            return False

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r""" log p(x | theta) """

        return self.likelihood(theta).log_prob(x)

    def sample(self, theta: torch.Tensor, shape: torch.Size = ()) -> torch.Tensor:
        r""" x ~ p(x | theta) """

        return self.likelihood(theta).sample(shape)

    def joint(self, shape: torch.Size = ()) -> Tuple[torch.Tensor, torch.Tensor]:
        r""" (theta, x) ~ p(theta) p(x | theta) """

        theta = self.prior.sample(shape)
        x = self.sample(theta)

        return theta, x


class ParallelSampler(data.Dataset):
    def __init__(self, simulator: Simulator, samples: int, batch_size: int, num_workers: int = None):
        super().__init__()

        self.simulator = simulator
        self.samples = samples
        self.batch_size = batch_size

        if num_workers is None:
            num_workers = len(psutil.Process().cpu_affinity())
        self.num_workers = num_workers

    def __len__(self) -> int:
        return math.ceil(self.samples / self.batch_size)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        offset = max(0, (i + 1) * self.batch_size - self.samples)
        return self.simulator.joint((self.batch_size - offset,))

    @staticmethod
    def collate_fn(x: Any) -> Any:
        return x

    @staticmethod
    def worker_init_fn(n: int) -> None:
        seed = torch.initial_seed() % 2 ** 32
        random.seed(seed)
        np.random.seed(seed)

    def __iter__(self):
        yield from data.DataLoader(
            self,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn,
        )
