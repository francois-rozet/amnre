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
        batch_size: int = 2 ** 10,  # 1024
    ):
        super().__init__()

        self.simulator = simulator
        self.prior = self.simulator.prior
        self.batch_shape = (batch_size,)

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        while True:
            theta, x = self.simulator.sample(self.batch_shape)
            theta_prime = self.prior.sample(self.batch_shape)

            if hasattr(self.simulator, 'noise'):
                x = x + self.simulator.noise(self.batch_shape)

            yield theta, theta_prime, x  # (theta, theta', x)


class OfflineLTEDataset(data.IterableDataset):
    r"""Offline Likelihood-To-Evidence Dataset"""

    def __init__(
        self,
        filename: str,
        chunk_size: str = 2 ** 16,  # 65536
        batch_size: int = 2 ** 10,  # 1024
        device: str = 'cpu',
    ):
        super().__init__()

        self.f = h5py.File(filename, 'r')

        self.chunks = [
            slice(i, min(i + chunk_size, len(self)))
            for i in range(0, len(self), chunk_size)
        ]

        self.batch_size = batch_size
        self.device = device

    def __len__(self) -> int:
        return len(self.f['theta'])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = torch.from_numpy(self.f['theta'][idx])
        x = torch.from_numpy(self.f['x'][idx])

        if 'noise' in self.f:
            noise = torch.from_numpy(self.f['noise'][idx])
            x = x + noise

        return theta, x

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        random.shuffle(self.chunks)

        for chunk in self.chunks:
            # Load
            theta_chunk = torch.from_numpy(self.f['theta'][chunk])
            x_chunk = torch.from_numpy(self.f['x'][chunk])

            # Shuffle
            order = torch.randperm(len(theta_chunk))
            theta_chunk, x_chunk = theta_chunk[order], x_chunk[order]

            # Noise
            if 'noise' in self.f:
                noise_chunk = torch.from_numpy(self.f['noise'][chunk])
                x_chunk = x_chunk + noise_chunk

            # CUDA
            if self.device == 'cuda':
                theta_chunk, x_chunk = theta_chunk.pin_memory(), x_chunk.pin_memory()

            # Split
            for theta, x in zip(
                theta_chunk.split(self.batch_size),
                x_chunk.split(self.batch_size),
            ):
                theta, x = theta.to(self.device), x.to(self.device)
                theta_prime = theta[torch.randperm(len(theta))]

                yield theta, theta_prime, x
