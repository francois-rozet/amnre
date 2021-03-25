#!/usr/bin/env python

import h5py
import random
import torch
import torch.utils.data as data

from typing import Tuple

from .simulators import Distribution, Simulator


class OnlineLTEDataset(data.IterableDataset):
    r"""Online Likelihood-To-Evidence Dataset"""

    def __init__(
        self,
        simulator: Simulator,
        batch_size: int = 1024,
    ):
        super().__init__()

        self.simulator = simulator
        self.prior = self.simulator.prior
        self.batch_shape = (batch_size,)

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        while True:
            theta, x = self.simulator.sample(self.batch_shape)
            theta_prime = self.prior.sample(self.batch_shape)

            yield theta, theta_prime, x  # (theta, theta', x)


class OfflineLTEDataset(data.IterableDataset):
    r"""Offline Likelihood-To-Evidence Dataset"""

    def __init__(
        self,
        filename: str,
        prior: Distribution,
        batch_size: int = 1024,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.f = h5py.File(filename, 'r')
        self.chunks = list(chunk[0] for chunk in self.f['x'].iter_chunks())

        self.prior = prior
        self.pin = self.prior.sample().is_cuda

        self.batch_shape = (batch_size,)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = torch.from_numpy(self.f['theta'][idx])
        x = torch.from_numpy(self.f['x'][idx])

        return theta, x

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        random.shuffle(self.chunks)

        for chunk in self.chunks:
            theta_chunk = torch.tensor(self.f['theta'][chunk])
            x_chunk = torch.tensor(self.f['x'][chunk])

            if self.pin:
                theta_chunk, x_chunk = theta_chunk.pin_memory(), x_chunk.pin_memory()

            order = torch.randperm(chunk.stop - chunk.start)

            for idx in order.split(self.batch_shape[0]):
                theta_prime = self.prior.sample(self.batch_shape)
                theta, x = theta_chunk[idx], x_chunk[idx]
                theta, x = theta.to(theta_prime), x.to(theta_prime)

                yield theta, theta_prime, x
