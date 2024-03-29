#!/usr/bin/env python

import h5py
import numpy as np
import torch
import torch.utils.data as data

from torch.distributions import Distribution
from typing import Tuple

from .simulators import Simulator


class OnlineDataset(data.IterableDataset):
    r"""Online Dataset"""

    def __init__(
        self,
        simulator: Simulator,
        batch_size: int = 2 ** 10,  # 1024
    ):
        super().__init__()

        self.simulator = simulator

        self.prior = self.simulator.prior
        self.batch_shape = (batch_size,)

    def __iter__(self):  # -> Tuple[torch.Tensor, torch.Tensor]
        while True:
            theta, x = self.simulator.joint(self.batch_shape)

            yield theta, x


class OfflineDataset(data.IterableDataset):
    r"""Offline Dataset"""

    def __init__(
        self,
        filename: str,  # H5
        chunk_size: str = 2 ** 18,  # 262144
        batch_size: int = 2 ** 10,  # 1024
        device: str = 'cpu',
        shuffle: bool = True,
        live: callable = None,
    ):
        super().__init__()

        self.f = h5py.File(filename, 'r')

        self.chunks = [
            slice(i, min(i + chunk_size, len(self)))
            for i in range(0, len(self), chunk_size)
        ]

        self.batch_size = batch_size
        self.device = device

        self.rng = np.random.default_rng()
        self.shuffle = shuffle

        if 'mu' in self.f:
            self.mu = torch.from_numpy(self.f['mu'][:]).to(device)
        else:
            self.mu = None

        if 'sigma' in self.f:
            self.isigma = torch.from_numpy(self.f['sigma'][:] ** -1).to(device)
        else:
            self.isigma = None

        self.live = live

    def __len__(self) -> int:
        return len(self.f['x'])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if 'theta' in self.f:
            theta = self.f['theta'][idx]
            theta = torch.from_numpy(theta).to(self.device)
        else:
            theta = None

        x = self.f['x'][idx]
        x = torch.from_numpy(x).to(self.device)

        if self.live is not None:
            x = self.live(theta, x)

        return theta, self.normalize(x)

    def __iter__(self):  # -> Tuple[torch.Tensor, torch.Tensor]
        if self.shuffle:
            self.rng.shuffle(self.chunks)

        for chunk in self.chunks:
            # Load
            theta_chunk, x_chunk = self.f['theta'][chunk], self.f['x'][chunk]

            ## Shuffle
            if self.shuffle:
                order = self.rng.permutation(len(x_chunk))
                theta_chunk, x_chunk = theta_chunk[order], x_chunk[order]

            # CUDA
            theta_chunk, x_chunk = torch.from_numpy(theta_chunk), torch.from_numpy(x_chunk)

            if self.device == 'cuda':
                theta_chunk, x_chunk = theta_chunk.pin_memory(), x_chunk.pin_memory()

            # Batches
            for theta, x in zip(
                theta_chunk.split(self.batch_size),
                x_chunk.split(self.batch_size),
            ):
                theta, x = theta.to(self.device), x.to(self.device)

                if self.live is not None:
                    x = self.live(theta, x)

                yield theta, self.normalize(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mu is not None:
            x = x - self.mu

        if self.isigma is not None:
            x = x * self.isigma

        return x


class LTEDataset(data.IterableDataset):
    r"""Likelihood-To-Evidence (LTE) dataset"""

    def __init__(self, dataset: data.IterableDataset, prior: Distribution = None, shift: int = 1):
        super().__init__()

        self.dataset = dataset
        self.prior = prior
        self.shift = shift

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):  # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        for theta, x in self.dataset:
            if self.prior is None:
                theta_prime = torch.roll(theta, self.shift, 0)
            else:
                theta_prime = self.prior.sample(theta.shape[:-1])

            yield theta, theta_prime, x
