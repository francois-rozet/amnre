#!/usr/bin/env python

import random
import torch
import torch.utils.data as data

from typing import Tuple

from simulators import Simulator


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

        self.all = torch.ones_like(self.simulator.prior.sample()).bool()

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        while True:
            theta, x = self.simulator.sample(self.batch_shape)
            theta_prime = self.simulator.prior.sample(self.batch_shape)

            # (theta, x)
            yield theta, x, self.all

            # (theta', x)
            yield theta_prime, x, ~self.all
