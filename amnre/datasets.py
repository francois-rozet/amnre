#!/usr/bin/env python

import h5py
import numpy as np
import torch
import torch.utils.data as data

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
        self.noisy = hasattr(self.simulator, 'noise')

        self.prior = self.simulator.prior
        self.batch_size = batch_size

    @property
    def batch_shape(self) -> torch.Size:
        return (self.batch_size,)

    def __iter__(self): # -> Tuple[torch.Tensor, torch.Tensor]
        while True:
            theta, x = self.simulator.sample(self.batch_shape)

            if self.noisy:
                x = x + self.simulator.noise(self.batch_shape)

            yield theta, x  # (theta, x)


class OfflineDataset(data.IterableDataset):
    r"""Offline Dataset"""

    def __init__(
        self,
        filename: str,  # H5
        chunk_size: str = 2 ** 18,  # 262144
        batch_size: int = 2 ** 10,  # 1024
        device: str = 'cpu',
    ):
        super().__init__()

        self.f = h5py.File(filename, 'r')
        self.noisy = 'noise' in self.f

        self.chunks = [
            slice(i, min(i + chunk_size, len(self)))
            for i in range(0, len(self), chunk_size)
        ]

        self.batch_size = batch_size
        self.device = device

        if 'mu' in self.f:
            self.mu = torch.from_numpy(self.f['mu'][:]).to(device)
        else:
            self.mu = None

        if 'sigma' in self.f:
            self.isigma = torch.from_numpy(self.f['sigma'][:]).to(device) ** -1
        else:
            self.isigma = None

    def __len__(self) -> int:
        return len(self.f['x'])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.noisy:
            x = self.f['x'][idx] + self.f['noise'][idx]
        else:
            x = self.f['x'][idx]

        x = torch.from_numpy(x).to(self.device)

        if 'theta' in self.f:
            theta = self.f['theta'][idx]
            theta = torch.from_numpy(theta).to(self.device)
        else:
            theta = None

        return theta, self.normalize(x)

    def __iter__(self): # -> Tuple[torch.Tensor, torch.Tensor]
        np.random.shuffle(self.chunks)

        for chunk in self.chunks:
            # Load
            theta_chunk, x_chunk = self.f['theta'][chunk], self.f['x'][chunk]

            ## Shuffle
            order = np.random.permutation(len(x_chunk))
            theta_chunk, x_chunk = theta_chunk[order], x_chunk[order]

            ## Noise
            if self.noisy:
                noise_chunk = np.random.permutation(self.f['noise'][chunk])
                x_chunk = x_chunk + noise_chunk

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

                yield theta, self.normalize(x)

    def normalize(self, x: torch.Tensor):
        if self.mu is not None:
            x = x - self.mu

        if self.isigma is not None:
            x = x * self.isigma

        return x


class LTEDataset(data.IterableDataset):
    r"""Likelihood-To-Evidence (LTE) dataset"""

    def __init__(self, dataset: data.IterableDataset):
        super().__init__()

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self): # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        for theta, x in self.dataset:
            theta_prime = theta[torch.randperm(len(theta))]

            yield theta, theta_prime, x
