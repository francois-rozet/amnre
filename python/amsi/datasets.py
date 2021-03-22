#!/usr/bin/env python

import random
import torch
import torch.utils.data as data

from typing import Tuple

from .simulators import Simulator


class LTEDataset(data.IterableDataset):
    r"""Likelihood-To-Evidence Dataset"""

    def __init__(
        self,
        simulator: Simulator,
        batch_size: int = 256,
    ):
        super().__init__()

        self.simulator = simulator
        self.batch_shape = (batch_size,)

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        while True:
            theta, x = self.simulator.sample(self.batch_shape)
            theta_prime = self.simulator.prior.sample(self.batch_shape)

            yield theta, theta_prime, x  # (theta, theta', x)
