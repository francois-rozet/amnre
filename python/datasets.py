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
        masks: torch.BoolTensor = None,
        batch_size: int = 256,
    ):
        super().__init__()

        self.simulator = simulator
        self.batch_size = batch_size

        if masks is None:
            self.masks = torch.ones_like(
                self.simulator.prior.sample()
            ).bool().unsqueeze(0)
        else:
            self.masks = masks

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        while True:
            theta, x = self.simulator.sample((self.batch_size,))
            theta_prime = self.simulator.prior.sample((self.batch_size,))

            mask = random.choice(self.masks)

            # (theta_a, theta_b', x)
            yield mask * theta + ~mask * theta_prime, x, mask

            # (theta_a', theta_b, x)
            yield ~mask * theta + mask * theta_prime, x, ~mask
