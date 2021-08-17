#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn

from typing import List


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

    def __init__(self, size: int, lam: float = 1., filtr: torch.BoolTensor = None):
        super().__init__()

        self.size = size
        self.lam = lam

        self.rng = np.random.default_rng()

        indices = torch.arange(self.size)
        if filtr is not None:
            indices = indices[filtr]

        self.register_buffer('indices', indices)

    @property
    def device(self) -> torch.device:
        return self.indices.device

    def forward(self, shape: torch.Size = ()) -> torch.BoolTensor:
        masks = torch.zeros(shape + (self.size,), dtype=bool, device=self.device)

        k = 1 + self.rng.poisson(self.lam)
        length = len(self.indices)

        if k < length:
            idx = torch.rand(shape + (length,), device=self.device)
            idx = torch.argsort(idx, dim=-1)[..., :k]
            idx = torch.take(self.indices, idx)

            masks.scatter_(-1, idx, True)
        else:
            masks[..., self.indices] = True

        return masks


def bit_repr(integers: torch.LongTensor, powers: torch.LongTensor) -> torch.BoolTensor:
    r"""Bit representation of integers"""

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


def list2masks(strings: List[str], size: int, filtr: str = None) -> torch.BoolTensor:
    if not strings:
        return torch.zeros((0, size), dtype=bool)

    masks = []

    every = enumerate_masks(size)

    if filtr is not None:
        filtr = str2mask(filtr)
        filtr = ~torch.any(every * ~filtr, dim=-1)
        every = every[filtr]

    sizes = every.sum(dim=1)

    for s in strings:
        if s.startswith('='):
            s = int(s[1:])
            masks.append(every[sizes == s])
        else:
            mask = str2mask(s)[:size]
            masks.append(mask.unsqueeze(0))

    return torch.cat(masks)
