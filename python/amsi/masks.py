#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn


class SelectionMask(nn.Module):
    r"""Selection mask sampler"""

    def __init__(self, masks: torch.BoolTensor):
        super().__init__()

        self.register_buffer('masks', masks)

    def forward(self, shape: torch.Size = ()) -> torch.BoolTensor:
        idx = torch.randint(len(self.masks), shape)
        return self.masks[idx]


class UniformMask(nn.Module):
    r"""Uniform mask sampler"""

    def __init__(self, size: int):
        super().__init__()

        self.size = size
        self.register_buffer('powers', 2 ** torch.arange(size))

    def forward(self, shape: torch.Size = ()) -> torch.BoolTensor:
        integers = torch.randint(1, 2 ** self.size, shape)
        return bit_repr(integers, self.powers)


class PoissonMask(nn.Module):
    r"""Poisson mask sampler"""

    def __init__(self, size: int, lam: float = 2.):
        super().__init__()

        self.size = size
        self.lam = lam

        self.register_buffer('powers', 2 ** torch.arange(self.size))

    def forward(self, shape: torch.Size = ()) -> torch.BoolTensor:
        k = 1 + np.random.poisson(self.lam)

        idx = torch.randint(self.size, shape + (k,))
        integers = np.bitwise_or.reduce(2 ** idx, axis=-1)

        return bit_repr(integers, self.powers)


def bit_repr(integers: torch.LongTensor, powers: torch.LongTensor) -> torch.BoolTensor:
    return integers.to(powers).unsqueeze(-1).bitwise_and(powers) != 0


def enumerate_masks(size: int) -> torch.BoolTensor:
    r"""Enumerate all possible masks"""

    powers = 2 ** torch.arange(size)
    integers = torch.arange(1, 2 ** size)

    return bit_repr(integers, powers)


def mask2str(mask: torch.BoolTensor) -> str:
    return ''.join('1' if b else '0' for b in mask)


def str2mask(s: str) -> torch.BoolTensor:
    return torch.tensor([c == '1' for c in s])
