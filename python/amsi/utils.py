#!/usr/bin/env python

import torch


def enumerate_masks(size: int) -> torch.BoolTensor:
    r"""Enumerate all possible masks"""

    powers = 2 ** torch.arange(size)
    integers = torch.arange(1, 2 ** size)
    masks = integers.unsqueeze(1).bitwise_and(powers) != 0

    return masks


def mask2str(mask: torch.BoolTensor) -> str:
    return ''.join('1' if b else '0' for b in mask)


def str2mask(s: str) -> torch.BoolTensor:
    return torch.tensor([c == '1' for c in s])
